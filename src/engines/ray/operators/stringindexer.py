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


def run_stringindexer_with_ray_data(ray_dataset: ray.data.Dataset, spec: OperatorSpec):
    """
    使用Ray官方LabelEncoder API - 核心实现（仅用于benchmark）
    
    Args:
        ray_dataset: Ray Dataset（必须是Ray Dataset，不接受pandas）
        spec: 算子规格
        
    Returns:
        Ray Dataset
    """
    # 强校验：必须是Ray Dataset
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(f"Expected ray.data.Dataset, got {type(ray_dataset)}. "
                       f"Use run_stringindexer() wrapper for pandas input.")
    
    # 获取输入列和输出列
    input_cols = spec.input_cols
    output_cols = spec.output_cols
    
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
    
    return result


def run_stringindexer(input_data, spec: OperatorSpec):
    """
    便利wrapper - 支持pandas或Ray Dataset输入
    
    注意：benchmark代码应该直接调用run_stringindexer_with_ray_data
    """
    if isinstance(input_data, ray.data.Dataset):
        return run_stringindexer_with_ray_data(input_data, spec)
    elif isinstance(input_data, pd.DataFrame):
        ray_dataset = ray.data.from_pandas(input_data)
        return run_stringindexer_with_ray_data(ray_dataset, spec)
    else:
        raise TypeError(f"Expected ray.data.Dataset or pd.DataFrame, got {type(input_data)}")
