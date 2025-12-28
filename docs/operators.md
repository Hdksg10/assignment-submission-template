# 算子规格说明

## 算子分类

### 数值预处理 (Numerical Preprocessing)

#### StandardScaler
- **功能**: 标准化数值特征 (z-score标准化)
- **输入**: 数值列
- **输出**: 标准化后的数值列
- **参数**:
  - `with_mean`: 是否中心化 (默认: True)
  - `with_std`: 是否标准化 (默认: True)
- **Spark实现**: `StandardScaler`
- **Ray实现**: 使用Ray Data map_batches + scikit-learn

#### MinMaxScaler
- **功能**: 最小最大标准化
- **输入**: 数值列
- **输出**: [0,1]范围内的数值列
- **参数**:
  - `min`: 最小值 (默认: 0.0)
  - `max`: 最大值 (默认: 1.0)
- **Spark实现**: `MinMaxScaler`
- **Ray实现**: 使用Ray Data map_batches + scikit-learn

#### Imputer
- **功能**: 缺失值填充
- **输入**: 包含缺失值的数值列
- **输出**: 填充后的数值列
- **参数**:
  - `strategy`: 填充策略 ('mean', 'median', 'most_frequent', 'constant')
  - `fill_value`: 常量填充值 (当strategy='constant'时)
- **Spark实现**: `Imputer`
- **Ray实现**: 使用Ray Data map_batches + pandas/scikit-learn

### 类别预处理 (Categorical Preprocessing)

#### StringIndexer
- **功能**: 字符串到数字的映射
- **输入**: 类别字符串列
- **输出**: 数字编码列
- **参数**:
  - `handle_invalid`: 无效值的处理方式 ('error', 'skip', 'keep')
- **Spark实现**: `StringIndexer`
- **Ray实现**: 使用Ray Data map_batches + sklearn.preprocessing.LabelEncoder

#### OneHotEncoder
- **功能**: 独热编码
- **输入**: 数字编码列
- **输出**: 独热编码的多列
- **参数**:
  - `drop_last`: 是否丢弃最后一列避免多重共线性 (默认: True)
- **Spark实现**: `OneHotEncoder`
- **Ray实现**: 使用Ray Data map_batches + sklearn.preprocessing.OneHotEncoder

### 文本预处理 (Text Preprocessing)

#### Tokenizer
- **功能**: 文本分词
- **输入**: 文本字符串列
- **输出**: 分词后的数组列
- **参数**:
  - `pattern`: 分词模式 (正则表达式)
- **Spark实现**: `Tokenizer`
- **Ray实现**: 使用Ray Data map_batches + NLTK或自定义分词器

#### HashingTF
- **功能**: 特征哈希向量化
- **输入**: 词汇数组列
- **输出**: 哈希特征向量
- **参数**:
  - `num_features`: 特征维度 (默认: 2^18)
- **Spark实现**: `HashingTF`
- **Ray实现**: 使用Ray Data map_batches + sklearn.feature_extraction.text.HashingVectorizer

#### IDF (Inverse Document Frequency)
- **功能**: 逆文档频率转换
- **输入**: 词频向量列
- **输出**: TF-IDF向量
- **参数**:
  - `min_doc_freq`: 最小文档频率 (默认: 1)
- **Spark实现**: `IDF`
- **Ray实现**: 使用Ray Data map_batches + sklearn.feature_extraction.text.TfidfTransformer

## 实现状态

### 已实现算子
- ✅ StandardScaler (Spark + Ray)

### 开发中算子
- 🔄 StringIndexer (计划中)
- 🔄 OneHotEncoder (计划中)

### 待实现算子
- ⏳ MinMaxScaler
- ⏳ Imputer
- ⏳ Tokenizer
- ⏳ HashingTF
- ⏳ IDF

## 算子规格定义

每个算子需要定义以下规格：

```python
OperatorSpec(
    name="StandardScaler",
    input_cols=["feature1", "feature2"],
    output_cols=["feature1_scaled", "feature2_scaled"],
    params={
        "with_mean": True,
        "with_std": True
    },
    description="Standardize features by removing the mean and scaling to unit variance"
)
```

## 添加新算子流程

1. **定义规格**: 在 `src/bench/operator_spec.py` 中添加算子规格
2. **Spark实现**: 在 `src/engines/spark/operators/` 中实现对应函数
3. **Ray实现**: 在 `src/engines/ray/operators/` 中实现对应函数
4. **更新文档**: 在本文档中添加算子说明
5. **添加测试**: 在 `tests/` 中添加相应的测试用例
6. **验证一致性**: 确保两个引擎输出结果的一致性
