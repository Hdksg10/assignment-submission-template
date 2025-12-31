# 基准测试触发机制改进

## 改造目标

降低评测框架带来的系统性偏差：
1. **Spark**: `count()` 会触发 column pruning，导致 transform 没真正被计算
2. **Ray**: `materialize()` 把"物化成本"掺进来，不是算子本体时间

## 核心改进

### 1. 新的触发机制：`force_execute()`

**原则**: 触发执行必须"依赖输出列的值"，避免 Spark Catalyst 把输出列计算裁剪掉

#### Spark 实现
- 使用聚合 checksum（`sum()` 等）强制计算输出列
- 数值列：`sum(col)`
- 字符串列：`sum(length(col))`
- 复杂类型：`sum(length(cast(string)))`
- 确保 Spark Catalyst 不会优化掉 transform 计算

#### Ray 实现
- 使用 per-batch checksum + 小量回传
- **不进行 materialize()**，避免物化/缓存写入开销
- checksum 依赖输出列，保证 map/transform 真被执行

### 2. 统一接口

```python
from bench.materialize import force_execute

# 使用输出列触发
force_execute(output, engine="spark", cols=spec.output_cols)
force_execute(output, engine="ray", cols=spec.output_cols)
```

### 3. 自动获取输出列

`ExperimentRunner` 现在自动从 `spec.output_cols` 获取触发列：
- 优先使用 `spec.output_cols`
- Fallback 到 `spec.input_cols`（不推荐，会警告）
- 如果都为空，fallback 到 `count()`（会警告）

### 4. 统一使用 `perf_counter()`

所有性能测量统一使用 `time.perf_counter()`，减少微基准噪声：
- `MetricsCollector.start_measurement()` / `end_measurement()`
- `PerformanceOptimizedTimer`（已经使用）

## 文件变更

### 修改的文件

1. **`src/bench/materialize.py`**
   - 重写：实现 `force_execute()` 统一接口
   - Spark: 聚合 checksum 触发
   - Ray: per-batch checksum，不 materialize
   - 保留旧接口（deprecated）用于向后兼容

2. **`src/bench/metrics.py`**
   - `MetricsCollector`: 使用 `perf_counter()` 替代 `time.time()`
   - `ExperimentRunner.run_experiment()`: 
     - 移除 `materialize_func` 参数
     - 自动从 `spec.output_cols` 获取触发列
     - 使用 `force_execute()` 替代 `materialize_func()`

3. **`src/bench/cli.py`**
   - 移除 `materialize_func` 参数的传递
   - `ExperimentRunner` 现在自动处理触发

4. **`src/bench/pipeline_executor.py`**
   - `execute_pipeline()`: 使用 `force_execute()` 替代 `count()`
   - `execute_pipeline_with_detailed_metrics()`: 使用 `force_execute()` 替代 `count()`
   - 从 `step_spec.output_cols` 获取触发列

### 新增文件

5. **`tests/test_benchmark_trigger.py`**
   - 防回归测试：确保 Spark 触发不会被 column pruning 裁剪
   - 测试多个输出列的情况
   - 测试 Ray 不进行 materialize

## 技术细节

### Spark 防 Column Pruning

**问题**: `count()` 只计算行数，Spark Catalyst 可能优化掉输出列的计算

**解决**: 使用聚合表达式依赖输出列的值
```python
# 数值列
exprs.append(F.sum(col).cast("double"))

# 字符串列  
exprs.append(F.sum(F.length(col)).cast("double"))

# 复杂类型
exprs.append(F.sum(F.length(col.cast("string"))).cast("double"))
```

### Ray 避免 Materialize 开销

**问题**: `materialize()` 会把数据写入对象存储/缓存，这不是算子本体时间

**解决**: 使用 per-batch checksum，只回传小量数据
```python
def batch_checksum(df: pd.DataFrame) -> pd.DataFrame:
    # 计算每批的 checksum
    h = pd.util.hash_pandas_object(sub, index=False).astype("uint64")
    chk = int(h.sum() & np.uint64(0xFFFFFFFFFFFFFFFF))
    return pd.DataFrame({"__chk": [chk]})

# 只回传每个 block 1 行
parts = chk_ds.take_all()
```

## 向后兼容性

- 保留 `materialize_spark()` 和 `materialize_ray()` 函数（标记为 deprecated）
- 旧代码仍可使用，但会显示警告
- 建议迁移到 `force_execute()`

## 验证

运行防回归测试：
```bash
python -m pytest tests/test_benchmark_trigger.py -v
```

测试确保：
1. Spark 触发不会被 column pruning 裁剪
2. Ray 不进行 materialize
3. 多个输出列正确处理
4. Fallback 行为正确

## 使用示例

### 单个算子

```python
from bench.metrics import ExperimentRunner
from bench.operator_spec import get_operator_spec

spec = get_operator_spec("StandardScaler")
runner = ExperimentRunner(repeats=3, warmup=True)

result = runner.run_experiment(
    engine="spark",
    operator="StandardScaler",
    dataset_path="data.csv",
    operator_func=run_standardscaler,
    input_profile_df=df,
    spark=spark,
    input_df=spark_df,
    spec=spec  # ← force_execute 会自动使用 spec.output_cols
)
```

### Pipeline

```python
from bench.pipeline_executor import HighPerformancePipelineExecutor

executor = HighPerformancePipelineExecutor(engine="spark", spark_session=spark)
final_df, elapsed = executor.execute_pipeline(
    steps=pipeline_config.steps,
    input_df=spark_df,
    measure_performance=True
)
# ← 自动使用最后一步的 output_cols 触发
```

## 性能影响

### 改进前
- Spark: `count()` → 可能 column pruning，transform 被优化掉 ❌
- Ray: `materialize() + count()` → 物化开销被计入 ❌

### 改进后
- Spark: 聚合 checksum → 强制计算输出列 ✅
- Ray: per-batch checksum → 不 materialize，只回传小量数据 ✅

## 关键优势

1. **准确性**: 确保 transform 真正被执行
2. **纯净性**: Ray 不包含物化开销
3. **自动化**: 自动从 spec 获取输出列
4. **防回归**: 测试确保不会回退到错误实现

