# Ray/Spark 算子列一致性修复

## 问题描述

`test_operator_consistency` 测试失败，错误信息："列名不一致"

### 根本原因

**Spark 实现**:
- 保留所有原始列 + 添加新的输出列
- 例如：`['x1', 'x2', 'cat', 'text']` → `['x1', 'x2', 'cat', 'text', 'x1_scaled', 'x2_scaled']`
- 总计：6列

**Ray 实现（修复前）**:
- Ray 的 `StandardScaler` 会**原地替换** input_cols
- 然后 `rename` 把列名改为 output_cols
- 例如：`['x1', 'x2', 'cat', 'text']` → `['x1_scaled', 'x2_scaled', 'cat', 'text']`
- 总计：4列（原始列丢失）

## 解决方案：最小化 Overhead 的实现

### 核心思想

只需 **1次额外的 `map_batches` 调用**（仅当 `input_cols != output_cols` 时）：

1. **在 transform 前**：复制 `input_cols` 到 `output_cols`
2. **StandardScaler 在 `output_cols` 上操作**（保留 `input_cols` 原始值）
3. 完成！

### 代码实现

```python
# StandardScaler 和 MinMaxScaler 的通用模式
if input_cols != output_cols:
    # 在transform前复制input_cols到output_cols
    def duplicate_columns(batch):
        for in_col, out_col in zip(input_cols, output_cols):
            batch[out_col] = batch[in_col].copy()
        return batch
    
    ray_dataset = ray_dataset.map_batches(duplicate_columns, batch_format="pandas")
    
    # Scaler在output_cols上操作（这样input_cols保留原始值）
    preprocessor = RayStandardScaler(columns=output_cols)
else:
    # 如果列名相同，直接在input_cols上操作
    preprocessor = RayStandardScaler(columns=input_cols)

fitted = preprocessor.fit(ray_dataset)
result = fitted.transform(ray_dataset)
```

### Overhead 分析

| 场景 | 额外 map_batches 调用 | 说明 |
|------|---------------------|------|
| `input_cols == output_cols` | 0 | 直接 transform，零 overhead |
| `input_cols != output_cols` | 1 | 只复制列，最小 overhead |

### 与其他方案对比

#### 方案 A：保存-恢复（2次 map_batches）
```python
# 1. 保存原始列
ray_dataset = ray_dataset.map_batches(save_original, ...)
# 2. Transform（修改input_cols）
result = fitted.transform(ray_dataset)
# 3. 恢复原始列
result = result.map_batches(restore_original, ...)
```
- Overhead: **2次 map_batches** ❌

#### 方案 B：复制后操作（1次 map_batches） ✅
```python
# 1. 复制列
ray_dataset = ray_dataset.map_batches(duplicate_columns, ...)
# 2. Transform（修改output_cols，input_cols未动）
result = fitted.transform(ray_dataset)
```
- Overhead: **1次 map_batches** ✅

## 已更新的算子

### ✅ StandardScaler
- 文件：`src/engines/ray/operators/standardscaler.py`
- 修复：使用复制列方案

### ✅ MinMaxScaler
- 文件：`src/engines/ray/operators/minmaxscaler.py`
- 修复：使用复制列方案

### ✅ StringIndexer
- 文件：`src/engines/ray/operators/stringindexer.py`
- 状态：Ray 的 `LabelEncoder` 本身就支持 `output_column` 参数，已经保留原始列
- 无需修改

### ⚠️ OneHotEncoder
- 文件：`src/engines/ray/operators/onehotencoder.py`
- 状态：需要验证 Ray 的 `OneHotEncoder` 行为
- 可能需要类似修复（待确认）

## 验证

运行测试脚本验证修复：

```bash
python test_consistency_fix.py
```

预期输出：
```
============================================================
测试 StandardScaler 列一致性
============================================================

Spark 实现
------------------------------------------------------------
输出列: ['x1', 'x2', 'cat', 'text', 'x1_scaled', 'x2_scaled']
列数: 6

Ray 实现
------------------------------------------------------------
输出列: ['x1', 'x2', 'cat', 'text', 'x1_scaled', 'x2_scaled']
列数: 6

一致性检查
------------------------------------------------------------
✓ 列名一致！
```

## 性能影响

### 计时边界保持正确

修复不影响计时边界，因为：
1. 额外的 `map_batches` 在算子内部，仍然在计时区间内
2. 与 Spark 的行为一致（Spark 也需要操作多个列）
3. 是算子本体逻辑的一部分，不是框架 overhead

### Overhead 最小化

- **最佳情况**（`input_cols == output_cols`）：0 次额外调用
- **一般情况**（`input_cols != output_cols`）：1 次额外调用
- 复制操作是内存拷贝，非常快（相比网络传输、序列化等）

## 相关文档

- [BENCHMARK_TIMING_CHANGES.md](BENCHMARK_TIMING_CHANGES.md) - 计时边界改进
- [TIMING_BOUNDARIES_GUIDE.md](TIMING_BOUNDARIES_GUIDE.md) - 使用指南

