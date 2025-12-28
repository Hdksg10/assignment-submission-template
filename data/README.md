# 数据管理规范

## 目录结构说明

- `data/raw/` - 原始数据集，只允许存放小样本数据用于测试
- `data/processed/` - 算子处理后的输出数据，运行时生成，不提交到Git

## 数据管理规则

### 小样本数据 (data/raw/)

- **允许提交**: 1k~10k 行的小数据集，用于单测和冒烟测试
- **文件格式**: CSV、Parquet等常见格式
- **命名规范**: `sample.csv`, `test_data.parquet` 等
- **用途**: 单元测试、冒烟测试、CI验证

### 大数据集处理

- **不提交到Git**: 大型数据集（>10k行）禁止提交
- **下载脚本**: 使用 `scripts/download_datasets.sh` 下载
- **校验机制**: 提供MD5校验和、行数、列名示例
- **存储位置**: 本地临时目录或云存储

### 处理结果 (data/processed/)

- **运行时生成**: 算子处理后的中间结果
- **不提交**: 所有内容都通过 `.gitignore` 排除
- **清理**: 可通过脚本定期清理

## 示例数据集

当前仓库包含以下小样本数据集：

### sample.csv
- **行数**: ~1000行
- **列**: x1, x2 (数值), cat (类别), text (文本)
- **用途**: 标准算子测试和对比基准

### 数据校验示例

```python
# 校验sample.csv
import pandas as pd
df = pd.read_csv('data/raw/sample.csv')
assert df.shape[0] == 1000  # 行数校验
assert list(df.columns) == ['x1', 'x2', 'cat', 'text']  # 列名校验
assert df.dtypes['x1'] == 'float64'  # 类型校验
```

## 数据下载流程

对于大型数据集：

1. 运行下载脚本: `bash scripts/download_datasets.sh <dataset_name>`
2. 脚本会自动校验文件完整性
3. 数据存放于临时目录，不进入版本控制

## 注意事项

- 所有测试用例应使用 `data/raw/` 中的小样本
- 性能基准测试可以使用下载的大型数据集
- 敏感数据或真实业务数据请勿提交
- 定期检查 `.gitignore` 确保临时数据不被意外提交
