# 简化方案：遵循 MLlib 标准行为

## 问题发现

当前实现中，Spark wrapper **过度包装**了 MLlib 的标准行为。

### Spark MLlib 标准行为
- ✅ 转换器输出到新列（`inputCol` → `outputCol`）
- ❌ **不保留原始列**（这是标准行为）

### 当前 wrapper 的问题
```python
# Spark wrapper (当前)
output_columns = existing_cols + output_cols  # ← 不必要的保留
final_df = scaled_df.select(*output_columns)
```

这导致：
1. 不符合 MLlib 标准语义
2. 额外的列浪费内存
3. Ray 需要模拟这个非标准行为

## 建议方案：简化为标准行为

### 原则
**遵循 Spark MLlib 的标准行为：只输出必要的列**

### Spark 实现（简化）

```python
def run_standardscaler(spark, input_df: DataFrame, spec: OperatorSpec) -> DataFrame:
    """简化版：遵循 MLlib 标准行为"""
    input_cols = spec.params.get("input_cols", spec.input_cols)
    output_cols = spec.params.get("output_cols", spec.output_cols)
    
    # 组装向量
    assembler = VectorAssembler(inputCols=input_cols, outputCol="_vector_features")
    assembled_df = assembler.transform(input_df)
    
    # 标准化
    scaler = StandardScaler(
        inputCol="_vector_features",
        outputCol="_scaled_features",
        withMean=spec.params.get("with_mean", True),
        withStd=spec.params.get("with_std", True)
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)
    
    # 拆分向量到输出列
    scaled_df = scaled_df.withColumn(
        "_scaled_array",
        vector_to_array(col("_scaled_features"))
    )
    
    for i, output_col in enumerate(output_cols):
        scaled_df = scaled_df.withColumn(output_col, col("_scaled_array")[i])
    
    # 关键改变：只保留必要的列
    # 方案1: 删除 input_cols（如果 input_cols != output_cols）
    if input_cols != output_cols:
        keep_cols = [c for c in input_df.columns if c not in input_cols] + output_cols
    else:
        keep_cols = [c for c in input_df.columns if c not in input_cols] + output_cols
    
    # 方案2: 更简单 - 只删除中间临时列
    keep_cols = [c for c in scaled_df.columns 
                 if not c.startswith('_')]  # 删除 _vector_features 等临时列
    
    return scaled_df.select(*keep_cols)
```

### Ray 实现（简化）

```python
def run_standardscaler_with_ray_data(ray_dataset: ray.data.Dataset, 
                                      spec: OperatorSpec):
    """简化版：遵循标准行为，不保留原始列"""
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(...)
    
    input_cols = spec.params.get("input_cols", spec.input_cols)
    output_cols = spec.params.get("output_cols", spec.output_cols)
    
    # 直接在 input_cols 上操作
    preprocessor = RayStandardScaler(columns=input_cols)
    fitted = preprocessor.fit(ray_dataset)
    result = fitted.transform(ray_dataset)
    
    # 如果需要重命名，简单重命名即可（不保留原始列）
    if input_cols != output_cols:
        rename_map = dict(zip(input_cols, output_cols))
        result = result.map_batches(
            lambda batch: batch.rename(columns=rename_map),
            batch_format="pandas"
        )
    
    return result
```

## 对比：三种方案

| 方案 | Spark Overhead | Ray Overhead | 语义 | 内存效率 |
|------|---------------|--------------|------|----------|
| **方案1: 保留原始列** | 复制列 | 复制列 (1次 map_batches) | 非标准 | ❌ 差 |
| **方案2: 标准行为** | 无 | 无（或1次rename） | ✅ 标准 | ✅ 好 |

## 优势

### 1. 符合 MLlib 标准语义
```python
# MLlib 标准用法
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
```
- 输入列名：`features`
- 输出列名：`scaledFeatures`
- **不保留** `features` 列（标准行为）

### 2. 更好的性能
- 不复制列数据
- 更少的内存占用
- Ray 端零额外 overhead（除非需要 rename）

### 3. 更清晰的语义
- `input_cols != output_cols`: 重命名列
- `input_cols == output_cols`: 原地替换

### 4. Pipeline 友好
```python
# 标准 Pipeline 用法
pipeline = Pipeline(stages=[
    StandardScaler(inputCol="x1", outputCol="x1_scaled"),
    MinMaxScaler(inputCol="x1_scaled", outputCol="x1_normalized")
])
```
- 不需要担心列名冲突
- 每个阶段消费上一阶段的输出

## 建议实施步骤

### 选项 A：完全遵循标准（推荐）

1. **修改 Spark wrapper**
   - 移除 `existing_cols + output_cols` 的保留逻辑
   - 如果 `input_cols != output_cols`，删除 input_cols

2. **恢复 Ray 简单实现**
   - 回到原地替换 + rename 的简单逻辑
   - 移除复制列的代码

3. **更新测试**
   - 测试期望的列数应该是：`原始列数 - len(input_cols) + len(output_cols)`

### 选项 B：可配置行为

添加参数控制是否保留原始列：
```python
spec.params.get("keep_original_cols", False)
```

但这增加了复杂度，不推荐。

## 结论

**建议采用标准行为（选项 A）**：
- ✅ 更符合 MLlib 语义
- ✅ 更好的性能
- ✅ 更简单的代码
- ✅ 更少的 overhead

**当前"保留原始列"的实现是过度包装，应该简化。**

