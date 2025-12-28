# 数据集说明

## 内置小样本数据集

### sample.csv

仓库自带的小样本数据集，用于测试和演示。

#### 数据特征
- **文件位置**: `data/raw/sample.csv`
- **行数**: 1000行
- **列数**: 4列
- **大小**: ~50KB

#### Schema定义

| 列名 | 类型 | 描述 | 示例值 |
|------|------|------|--------|
| x1 | float64 | 数值特征1 | 1.23 |
| x2 | float64 | 数值特征2 | -0.45 |
| cat | string | 类别特征 | 'A', 'B', 'C' |
| text | string | 文本特征 | 'sample text data' |

#### 数据分布
- **x1**: 正态分布 N(0, 1)，含少量缺失值
- **x2**: 均匀分布 U(-1, 1)
- **cat**: 三个类别，近似均匀分布
- **text**: 随机生成的短文本句子

#### 生成方式

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 1000

# 生成数值列
x1 = np.random.normal(0, 1, n_rows)
x2 = np.random.uniform(-1, 1, n_rows)

# 随机设置一些缺失值
mask = np.random.random(n_rows) < 0.05
x1[mask] = np.nan

# 生成类别列
categories = ['A', 'B', 'C']
cat = np.random.choice(categories, n_rows)

# 生成文本列
texts = [f"sample text {i}" for i in range(n_rows)]
text = np.random.choice(texts, n_rows)

# 创建DataFrame
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'cat': cat,
    'text': text
})

# 保存为CSV
df.to_csv('data/raw/sample.csv', index=False)
```

#### 校验信息

```python
# 数据完整性校验
import pandas as pd

df = pd.read_csv('data/raw/sample.csv')
assert df.shape == (1000, 4)
assert list(df.columns) == ['x1', 'x2', 'cat', 'text']
assert df['x1'].isna().sum() > 0  # 确保有缺失值
assert len(df['cat'].unique()) == 3  # 三个类别
```

## 大数据集下载

### 支持的数据集

#### 1. 房价预测数据集 (House Prices)
- **来源**: Kaggle House Prices: Advanced Regression Techniques
- **行数**: ~1500行 (训练集)
- **特征**: 80+ 数值和类别特征
- **下载命令**: `bash scripts/download_datasets.sh house_prices`

#### 2. 信用卡欺诈检测 (Credit Card Fraud)
- **来源**: Kaggle Credit Card Fraud Detection
- **行数**: ~280k行
- **特征**: 28个匿名特征 + 时间 + 金额
- **下载命令**: `bash scripts/download_datasets.sh credit_fraud`

#### 3. 新闻分类数据集 (20 Newsgroups)
- **来源**: scikit-learn内置数据集
- **行数**: ~20k行
- **特征**: 文本数据，20个类别
- **下载命令**: `bash scripts/download_datasets.sh 20newsgroups`

### 下载校验

每个数据集提供MD5校验和和基本统计信息：

```json
{
  "dataset": "house_prices",
  "md5": "abc123...",
  "rows": 1460,
  "columns": 81,
  "schema": {
    "SalePrice": "float64",
    "LotArea": "int64",
    ...
  }
}
```

## 数据集使用指南

### 测试用例
```python
# 使用小样本数据集进行测试
from src.bench.io import load_csv

df = load_csv('data/raw/sample.csv')
print(f"数据集形状: {df.shape}")
print(f"列名: {list(df.columns)}")
```

### 性能测试
```python
# 下载并使用大数据集进行性能测试
import subprocess
subprocess.run(['bash', 'scripts/download_datasets.sh', 'house_prices'])

# 加载大数据集
df = load_csv('/tmp/datasets/house_prices/train.csv')
```

### 自定义数据集

添加新的数据集需要：

1. 在 `scripts/download_datasets.sh` 中添加下载逻辑
2. 创建校验文件 (JSON格式)
3. 更新本文档说明
4. 测试数据加载和算子兼容性

## 数据质量保证

### 完整性校验
- 文件MD5校验
- 行数和列数验证
- 数据类型检查

### 兼容性测试
- 确保所有算子都能处理数据集
- 验证输出Schema一致性
- 检查性能测试的稳定性
