# 基准测试计时边界改进 - 实施总结

## 概述

本次改进确保了 Spark 和 Ray 基准测试的计时边界准确且一致，只测量算子本体执行时间，排除了框架初始化、数据转换和结果收集的开销。

## 主要变更

### 1. 新增统一的触发执行工具 (`src/bench/materialize.py`)

创建了两个轻量级函数来触发分布式执行：

- **`materialize_spark(df)`**: 使用 `df.count()` 在 JVM 内触发执行
- **`materialize_ray(ds)`**: 使用 `ds.materialize()` + `ds.count()` 触发执行

这些函数只触发计算，不会将数据收集到 driver。

### 2. 改造 `MetricsCollector` (`src/bench/metrics.py`)

**变更前**:
```python
collect_metrics(input_df, output_df, elapsed_seconds, ...)
```

**变更后**:
```python
collect_metrics(input_rows, input_cols, output_rows, output_cols, elapsed_seconds, ...)
```

现在直接接受行列数，不依赖 DataFrame 的 `.shape` 属性（Spark DF / Ray Dataset 不支持）。

### 3. 更新 `ExperimentRunner` (`src/bench/metrics.py`)

新增两个必需参数：

- **`input_profile_df`**: pandas DataFrame，仅用于获取输入数据的行列数
- **`materialize_func`**: 触发执行的函数（在计时区间内调用）

**计时区间现在包含**:
```python
start_measurement()
output = operator_func(...)
materialize_func(output)  # ← 在计时内触发执行
elapsed = end_measurement()
```

**预热运行也包含 materialize**，确保 Spark/Ray 都真正执行作业。

### 4. 改造 CLI (`src/bench/cli.py`)

#### Spark 路径
```python
# 计时外：初始化 + 数据转换
spark = get_spark("BenchmarkApp")
spark_df = spark.createDataFrame(df)

# 传入 Runner
runner.run_experiment(
    input_profile_df=df,           # pandas 用于 profile
    materialize_func=materialize_spark,
    input_df=spark_df,             # Spark DF 作为算子输入
    ...
)
```

#### Ray 路径
```python
# 计时外：初始化 + 数据转换
init_ray()
ray_ds = rd.from_pandas(df)

# 传入 Runner
runner.run_experiment(
    input_profile_df=df,           # pandas 用于 profile
    materialize_func=materialize_ray,
    input_df=ray_ds,               # Ray Dataset 作为算子输入
    ...
)
```

### 5. 重构 Ray 算子 (`src/engines/ray/operators/*.py`)

每个算子现在分为两个函数：

#### 核心实现（用于 benchmark）
```python
def run_standardscaler_with_ray_data(ray_dataset: ray.data.Dataset, spec: OperatorSpec):
    """只接受 Ray Dataset，用于 benchmark"""
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(...)
    # ... 算子逻辑
    return result
```

#### 便利 wrapper（用于测试）
```python
def run_standardscaler(input_data, spec: OperatorSpec):
    """支持 pandas 或 Ray Dataset"""
    if isinstance(input_data, ray.data.Dataset):
        return run_standardscaler_with_ray_data(input_data, spec)
    elif isinstance(input_data, pd.DataFrame):
        ray_dataset = ray.data.from_pandas(input_data)
        return run_standardscaler_with_ray_data(ray_dataset, spec)
```

**移除了**:
- `take(1)` 纠错逻辑（这是 action，会污染计时）
- 算子内部的 `from_pandas` 转换（现在由 CLI 在计时外完成）

**更新的算子**:
- `standardscaler.py`
- `minmaxscaler.py`
- `stringindexer.py`
- `onehotencoder.py`

### 6. 修复 Pipeline 执行器 (`src/bench/pipeline_executor.py`)

#### 6.1 `PipelineConfig.from_operator_names`
- **不再原地修改** `spec.input_cols`
- 为每个步骤创建新的 `OperatorSpec` 拷贝

#### 6.2 `execute_pipeline`
- 新增 `per_step_timing` 参数
- **默认行为**: 只构建变换链，最后触发一次 action
- **per_step_timing=True**: 每步都 cache/materialize + trigger（用于详细分析）

```python
# 默认：只在最后触发
for context in contexts:
    current_df = execute_operator(context, current_df)
# 最后统一触发
materialize_func(current_df)

# per_step_timing: 每步都触发（会有持久化开销）
for context in contexts:
    current_df = execute_operator(context, current_df)
    current_df = current_df.cache()  # Spark
    current_df.count()
```

### 7. 更新测试 (`tests/*.py`)

#### `test_operator_contracts.py`
- Ray 算子测试：添加 `.to_pandas()` 转换
- 更新 `test_metrics_collection`：使用新的 `collect_metrics` 签名

```python
# 变更前
result_df = run_standardscaler(test_df, spec)
assert "x1_scaled" in result_df.columns

# 变更后
result_ds = run_standardscaler(test_df, spec)
result_df = result_ds.to_pandas()
assert "x1_scaled" in result_df.columns
```

#### `test_smoke.py`
- 同样添加 `.to_pandas()` 转换

## 验收标准

### ✅ 计时边界正确
- **包含**: `operator_func(...)` + `materialize()`
- **不包含**: Spark/Ray 初始化、pandas→分布式数据结构转换、结果收集

### ✅ 触发真实作业
- **Spark**: 能看到 job 被提交（通过 `count()`）
- **Ray**: 能看到执行被触发（通过 `materialize()` + `count()`）

### ✅ 一致性
- 改变输入转换方式不会显著改变测得的"算子耗时"
- Spark 和 Ray 的计时口径一致

### ✅ Pipeline 优化
- 默认不会每步 `count()`（避免重复回放）
- 只在最后触发一次 action
- 如需 per-step timing，显式开启并使用 cache/materialize

## 使用示例

### 运行单个算子 benchmark
```bash
python -m src.bench.cli run \
  --engine spark \
  --operator StandardScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3
```

### 运行 pipeline benchmark
```bash
python -m src.bench.cli pipeline \
  --engine ray \
  --operators StandardScaler MinMaxScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3
```

## 技术细节

### 为什么使用 `count()` 而不是 `collect()`？
- `count()`: 只在 executor 上计算行数，不传输数据到 driver
- `collect()`: 会将所有数据传输到 driver，引入网络开销

### 为什么 Ray 需要 `materialize()` + `count()`？
- `materialize()`: 确保数据被物化到内存/磁盘
- `count()`: 触发执行（如果还没执行）
- 避免重复计算

### 为什么需要 `input_profile_df`？
- Spark DF 和 Ray Dataset 没有 `.shape` 属性
- 需要一个轻量的 pandas DataFrame 来获取行列数
- 不参与算子执行，只用于统计

## 后续工作

如需支持更多算子，请遵循以下模式：

1. 在 CLI 中，计时外完成数据转换
2. 传入 `input_profile_df` 和 `materialize_func`
3. Ray 算子使用 `_with_ray_data` 核心实现
4. 测试使用 wrapper 函数并添加 `.to_pandas()`

## 相关文件

- `src/bench/materialize.py` - 新增
- `src/bench/metrics.py` - 修改
- `src/bench/cli.py` - 修改
- `src/bench/pipeline_executor.py` - 修改
- `src/engines/ray/operators/*.py` - 修改
- `tests/test_operator_contracts.py` - 修改
- `tests/test_smoke.py` - 修改

