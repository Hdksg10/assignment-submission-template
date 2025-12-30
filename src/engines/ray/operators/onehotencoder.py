"""
Ray OneHotEncoder算子 - 使用Ray Data官方preprocessor API
"""
import pandas as pd
import ray.data
from ray.data.preprocessors import OneHotEncoder
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_onehotencoder_with_ray_data(ray_dataset: ray.data.Dataset, spec: OperatorSpec):
    """
    使用Ray官方OneHotEncoder API - 核心实现（仅用于benchmark）
    
    Args:
        ray_dataset: Ray Dataset（必须是Ray Dataset，不接受pandas）
        spec: 算子规格
        
    Returns:
        Ray Dataset
    """
    # 强校验：必须是Ray Dataset
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(f"Expected ray.data.Dataset, got {type(ray_dataset)}. "
                       f"Use run_onehotencoder() wrapper for pandas input.")
    
    # 获取输入列和输出列
    input_cols = spec.input_cols
    output_cols = spec.output_cols
    
    if _logger:
        _logger.info(f"OneHotEncoder: {input_cols} -> {output_cols}")
    
    # 使用Ray的OneHotEncoder preprocessor
    preprocessor = OneHotEncoder(columns=input_cols)
    fitted = preprocessor.fit(ray_dataset)
    result = fitted.transform(ray_dataset)
    
    return result


def run_onehotencoder(input_data, spec: OperatorSpec):
    """
    便利wrapper - 支持pandas或Ray Dataset输入
    
    注意：benchmark代码应该直接调用run_onehotencoder_with_ray_data
    """
    if isinstance(input_data, ray.data.Dataset):
        return run_onehotencoder_with_ray_data(input_data, spec)
    elif isinstance(input_data, pd.DataFrame):
        ray_dataset = ray.data.from_pandas(input_data)
        return run_onehotencoder_with_ray_data(ray_dataset, spec)
    else:
        raise TypeError(f"Expected ray.data.Dataset or pd.DataFrame, got {type(input_data)}")
