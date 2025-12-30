# 基准测试计时边界使用指南

## 快速开始

### 运行单个算子基准测试

```bash
# Spark
python -m src.bench.cli run \
  --engine spark \
  --operator StandardScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3

# Ray
python -m src.bench.cli run \
  --engine ray \
  --operator StandardScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3
```

### 运行 Pipeline 基准测试

```bash
# Spark Pipeline
python -m src.bench.cli pipeline \
  --engine spark \
  --operators StandardScaler MinMaxScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3

# Ray Pipeline
python -m src.bench.cli pipeline \
  --engine ray \
  --operators StandardScaler MinMaxScaler \
  --input data/raw/sample.csv \
  --output experiments/runs \
  --repeats 3
```

## 计时边界说明

### ✅ 计时区间包含
- 算子函数执行 (`operator_func(...)`)
- 分布式执行触发 (`materialize_func(output)`)

### ❌ 计时区间不包含
- Spark/Ray 初始化 (`get_spark()` / `init_ray()`)
- 数据转换 (`pandas → Spark DF` / `pandas → Ray Dataset`)
- 结果收集 (`toPandas()` / `to_pandas()`)
- Pipeline 框架的额外 action

## 编程接口

### 使用 ExperimentRunner

```python
from bench.metrics import ExperimentRunner
from bench.materialize import materialize_spark, materialize_ray
from bench.operator_spec import get_operator_spec
import pandas as pd

# 准备数据
df = pd.read_csv("data.csv")
spec = get_operator_spec("StandardScaler")

# === Spark 示例 ===
from engines.spark.session import get_spark
from engines.spark.operators import run_standardscaler

spark = get_spark("BenchmarkApp")
spark_df = spark.createDataFrame(df)  # 计时外转换

runner = ExperimentRunner(repeats=3, warmup=True)
result = runner.run_experiment(
    engine="spark",
    operator="StandardScaler",
    dataset_path="data.csv",
    operator_func=run_standardscaler,
    input_profile_df=df,              # pandas 用于 profile
    materialize_func=materialize_spark,  # 触发执行
    spark=spark,
    input_df=spark_df,                # Spark DF 作为输入
    spec=spec
)

# === Ray 示例 ===
import ray.data as rd
from engines.ray.runtime import init_ray
from engines.ray.operators import run_standardscaler

init_ray()
ray_ds = rd.from_pandas(df)  # 计时外转换

runner = ExperimentRunner(repeats=3, warmup=True)
result = runner.run_experiment(
    engine="ray",
    operator="StandardScaler",
    dataset_path="data.csv",
    operator_func=run_standardscaler,
    input_profile_df=df,              # pandas 用于 profile
    materialize_func=materialize_ray,   # 触发执行
    input_df=ray_ds,                  # Ray Dataset 作为输入
    spec=spec
)
```

### 使用 Pipeline Executor

```python
from bench.pipeline_executor import (
    PipelineConfig,
    OptimizedPipelineRunner
)

# 创建 pipeline 配置
config = PipelineConfig.from_operator_names(
    operator_names=["StandardScaler", "MinMaxScaler"],
    engine="spark"
)

# 准备数据（计时外）
spark = get_spark("PipelineApp")
spark_df = spark.createDataFrame(df)

# 运行 pipeline
runner = OptimizedPipelineRunner(
    engine="spark",
    repeats=3,
    warmup=True
)

result = runner.run_pipeline_experiment(
    steps=config.steps,
    input_df=spark_df,
    spark_session=spark
)

print(f"平均耗时: {result['avg_time']:.3f}s")
print(f"吞吐量: {result['throughput_rows_per_sec']:.2f} rows/s")
```

## 添加新算子

### 1. 实现 Ray 算子

```python
# src/engines/ray/operators/myoperator.py
import pandas as pd
import ray.data
from bench.operator_spec import OperatorSpec

def run_myoperator_with_ray_data(ray_dataset: ray.data.Dataset, 
                                  spec: OperatorSpec):
    """核心实现 - 仅用于 benchmark"""
    # 强校验
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(f"Expected ray.data.Dataset, got {type(ray_dataset)}")
    
    # 算子逻辑
    # ...
    
    return result

def run_myoperator(input_data, spec: OperatorSpec):
    """便利 wrapper - 用于测试"""
    if isinstance(input_data, ray.data.Dataset):
        return run_myoperator_with_ray_data(input_data, spec)
    elif isinstance(input_data, pd.DataFrame):
        ray_dataset = ray.data.from_pandas(input_data)
        return run_myoperator_with_ray_data(ray_dataset, spec)
    else:
        raise TypeError(f"Unexpected type: {type(input_data)}")
```

### 2. 更新 CLI

```python
# src/bench/cli.py
def run_single_experiment(args):
    # ...
    
    if args.engine == 'ray':
        import ray.data as rd
        from engines.ray.operators import run_myoperator
        from bench.materialize import materialize_ray
        
        init_ray()
        ray_ds = rd.from_pandas(df)  # 计时外
        
        if args.operator == 'MyOperator':
            runner = ExperimentRunner(repeats=args.repeats, warmup=args.warmup)
            result = runner.run_experiment(
                engine=args.engine,
                operator=args.operator,
                dataset_path=args.input,
                operator_func=run_myoperator,
                input_profile_df=df,
                materialize_func=materialize_ray,
                input_df=ray_ds,
                spec=spec
            )
```

### 3. 编写测试

```python
# tests/test_myoperator.py
def test_myoperator_ray():
    from engines.ray.operators import run_myoperator
    from bench.operator_spec import get_operator_spec
    
    spec = get_operator_spec("MyOperator")
    test_df = pd.DataFrame({"x": [1, 2, 3]})
    
    # 执行算子（返回 Ray Dataset）
    result_ds = run_myoperator(test_df, spec)
    
    # 转换为 pandas 验证
    result_df = result_ds.to_pandas()
    
    assert len(result_df) == len(test_df)
    # ... 其他断言
```

## 常见问题

### Q: 为什么需要 `input_profile_df`？
A: Spark DF 和 Ray Dataset 没有 `.shape` 属性。我们需要一个轻量的 pandas DataFrame 来获取输入数据的行列数，用于计算吞吐量等指标。

### Q: `materialize_func` 做什么？
A: 它触发分布式执行。对于 Spark 是 `count()`，对于 Ray 是 `materialize() + count()`。这些操作不会将数据收集到 driver，只在 executor 上执行。

### Q: Pipeline 为什么不每步 count？
A: 每步 count 会导致 Spark lineage 反复回放，放大 pipeline 时间。默认只在最后触发一次 action。如需每步计时，使用 `per_step_timing=True`。

### Q: 测试为什么要 `.to_pandas()`？
A: Ray 算子现在返回 Ray Dataset，测试需要转换为 pandas 才能使用 `.columns`、`.mean()` 等方法。

### Q: Warmup 包含 materialize 吗？
A: 是的。Warmup 现在执行 `operator_func + materialize_func`，确保 Spark/Ray 都真正执行作业，而不只是构建执行计划。

## 验证

运行验证脚本检查所有组件：

```bash
python verify_timing_boundaries.py
```

预期输出：
```
============================================================
基准测试计时边界验证
============================================================

✓ 检查 materialize 模块...
  ✓ materialize_spark 存在
  ✓ materialize_ray 存在

✓ 检查 MetricsCollector.collect_metrics 签名...
  ✓ 参数 'input_rows' 存在
  ...

✓ 所有检查通过！
```

## 相关文档

- [BENCHMARK_TIMING_CHANGES.md](BENCHMARK_TIMING_CHANGES.md) - 详细的变更说明
- [README.md](README.md) - 项目总体说明

