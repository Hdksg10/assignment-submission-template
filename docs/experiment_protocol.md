# 实验协议

## 实验目标

量化对比Spark MLlib与Ray在机器学习预处理算子上的：
- **功能一致性**: 输出结果的等价性
- **性能表现**: 执行时间和资源使用效率
- **可扩展性**: 数据规模增长时的表现

## 指标定义

### 性能指标

#### 1. Wall Time (总耗时)
- **定义**: 从数据加载到结果输出的总时间
- **单位**: 秒
- **测量方式**: 使用Python的`time.time()`记录开始和结束时间
- **统计**: 记录平均值、标准差、最小值、最大值

#### 2. Throughput (吞吐量)
- **定义**: 每秒处理的行数
- **计算**: `rows_processed / elapsed_seconds`
- **单位**: rows/second
- **用途**: 衡量处理速度的直观指标

#### 3. Memory Usage (内存使用)
- **定义**: 峰值内存使用量
- **单位**: MB
- **测量**: 使用`psutil`或系统工具监控
- **可选**: 仅在资源受限环境测试

### 一致性指标

#### 4. Output Schema Consistency (输出Schema一致性)
- **检查项**:
  - 列名完全匹配
  - 数据类型一致
  - 行数不变（对于不改变行数的算子）

#### 5. Statistical Consistency (统计一致性)
- **数值列**: 均值、标准差、分布统计的相似性
- **类别列**: 类别分布的一致性
- **阈值**: 允许1%的相对误差

## 实验控制

### 执行参数

#### 重复次数
- **默认值**: 3次
- **目的**: 减少随机性影响，获得稳定的统计结果
- **统计方法**: 取平均值，计算标准差

#### Warm-up运行
- **次数**: 1次（丢弃结果）
- **目的**: 排除JIT编译、缓存预热等一次性开销

#### 随机种子
- **设置**: 所有引擎使用相同种子确保可复现性
- **种子值**: 42（或其他固定值）

### 环境隔离

#### 进程隔离
- **要求**: 各引擎测试在独立进程中执行
- **目的**: 避免内存泄漏、状态污染等问题

#### 资源控制
- **CPU**: 限制使用的CPU核心数
- **内存**: 设置合理的内存限制
- **I/O**: 控制并发I/O操作

## 实验流程

### 单次运行流程

```python
def run_single_experiment(engine, operator, dataset, params):
    # 1. 初始化引擎
    if engine == 'spark':
        spark = init_spark_session()
    elif engine == 'ray':
        init_ray_runtime()

    # 2. 加载数据
    df = load_dataset(dataset)

    # 3. 预热运行（可选）
    if params.get('warmup', True):
        _ = run_operator(engine, operator, df.copy())

    # 4. 正式运行（多次重复）
    results = []
    for i in range(params.get('repeats', 3)):
        start_time = time.time()
        output_df = run_operator(engine, operator, df.copy())
        end_time = time.time()

        # 5. 收集指标
        metrics = collect_metrics(output_df, end_time - start_time)
        results.append(metrics)

    # 6. 汇总结果
    final_result = aggregate_results(results)
    return final_result
```

### 对比测试流程

```python
def run_comparison_experiment(operator, dataset):
    # 并行或顺序运行两个引擎
    spark_result = run_single_experiment('spark', operator, dataset)
    ray_result = run_single_experiment('ray', operator, dataset)

    # 一致性校验
    consistency_report = check_output_consistency(
        spark_result['output'],
        ray_result['output']
    )

    # 生成对比报告
    comparison = generate_comparison_report(
        spark_result, ray_result, consistency_report
    )

    return comparison
```

## 结果输出格式

### 单次运行结果 (JSON)

```json
{
  "experiment_id": "spark_standardscaler_20241226_143000",
  "timestamp": "2024-12-26T14:30:00Z",
  "git_commit": "abc123def",
  "engine": "spark",
  "operator": "StandardScaler",
  "dataset": {
    "path": "data/raw/sample.csv",
    "rows": 1000,
    "columns": 4
  },
  "params": {
    "repeats": 3,
    "warmup": true
  },
  "metrics": {
    "wall_time": {
      "mean": 1.23,
      "std": 0.05,
      "min": 1.18,
      "max": 1.29
    },
    "throughput": {
      "mean": 813.01,
      "std": 40.65
    },
    "output": {
      "rows": 1000,
      "columns": 4,
      "schema": {
        "x1_scaled": "double",
        "x2_scaled": "double"
      }
    }
  }
}
```

### 对比报告 (JSON)

```json
{
  "comparison_id": "standardscaler_comparison_20241226_143000",
  "operator": "StandardScaler",
  "dataset": "data/raw/sample.csv",
  "engines": ["spark", "ray"],
  "performance_comparison": {
    "wall_time_ratio": 1.45,  // Spark相对于Ray的倍数
    "throughput_ratio": 0.69   // Spark相对于Ray的倍数
  },
  "consistency_check": {
    "schema_match": true,
    "row_count_match": true,
    "statistical_similarity": {
      "x1_scaled": {
        "mean_diff": 0.001,
        "std_diff": 0.002
      },
      "x2_scaled": {
        "mean_diff": 0.0005,
        "std_diff": 0.001
      }
    }
  },
  "recommendations": [
    "Spark在小数据集上表现更佳",
    "Ray具有更好的扩展潜力"
  ]
}
```

## 质量保证

### 可复现性
- **环境记录**: Python版本、依赖包版本、系统信息
- **配置备份**: 完整的实验配置保存
- **随机种子**: 固定种子确保结果可复现

### 异常处理
- **超时控制**: 单次运行最大时间限制
- **错误记录**: 详细的错误信息和堆栈跟踪
- **降级策略**: 遇到错误时提供备选方案

### 结果验证
- **统计合理性**: 检查结果的统计特征是否合理
- **边界条件**: 验证极端情况下的表现
- **交叉验证**: 使用不同数据集验证结论的普适性
