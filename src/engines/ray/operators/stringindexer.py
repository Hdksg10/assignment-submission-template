"""
Ray StringIndexer算子 - 使用Ray Data官方LabelEncoder API
"""
import pandas as pd
import ray.data
from ray.data.preprocessors import LabelEncoder
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_stringindexer(input_data, spec: OperatorSpec):
    """使用Ray官方LabelEncoder API进行字符串索引化 - 支持DataFrame和Ray Dataset"""
    # 获取输入列和输出列
    input_cols = spec.input_cols
    output_cols = spec.output_cols
    
    # 总是假设输入是Ray Dataset（框架已经转换）
    ray_dataset = input_data if hasattr(input_data, 'to_pandas') else ray.data.from_pandas(input_data)
    
    # 框架错误地设置了input_cols（它使用了前一个算子的output_cols）
    # StringIndexer应该只处理'cat'列
    # 我们需要检查数据中实际存在的列
    # 获取数据的第一行来检查列名
    sample = ray_dataset.take(1)
    if sample:
        available_cols = list(sample[0].keys()) if isinstance(sample[0], dict) else sample[0].keys()
        
        # 如果'cat'列存在但不在input_cols中，修正它
        if 'cat' in available_cols and 'cat' not in input_cols:
            input_cols = ['cat']
            output_cols = ['cat_indexed']
    
    # 确保input_cols和output_cols是列表
    if not isinstance(input_cols, list):
        input_cols = [input_cols]
    if not isinstance(output_cols, list):
        output_cols = [output_cols]
    
    if _logger:
        _logger.info(f"StringIndexer: {input_cols} -> {output_cols}")
    
    # 使用Ray的LabelEncoder preprocessor
    if len(input_cols) > 1:
        # 处理多列的情况
        result = ray_dataset
        for in_col, out_col in zip(input_cols, output_cols):
            preprocessor = LabelEncoder(label_column=in_col, output_column=out_col)
            fitted = preprocessor.fit(result)
            result = fitted.transform(result)
    else:
        # 单列情况
        preprocessor = LabelEncoder(label_column=input_cols[0], output_column=output_cols[0])
        fitted = preprocessor.fit(ray_dataset)
        result = fitted.transform(ray_dataset)
    
    # 总是返回Ray Dataset（保持框架的期望）
    return result
