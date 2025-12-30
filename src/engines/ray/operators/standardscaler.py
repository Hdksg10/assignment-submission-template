"""
Ray StandardScaler算子 - 使用Ray Data官方preprocessor API
"""
import pandas as pd
import ray.data
from ray.data.preprocessors import StandardScaler as RayStandardScaler
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_standardscaler_with_ray_data(ray_dataset: ray.data.Dataset, spec: OperatorSpec):
    """
    使用Ray官方StandardScaler API - 核心实现（仅用于benchmark）
    
    Args:
        ray_dataset: Ray Dataset（必须是Ray Dataset，不接受pandas）
        spec: 算子规格
        
    Returns:
        Ray Dataset
    """
    # 强校验：必须是Ray Dataset
    if not isinstance(ray_dataset, ray.data.Dataset):
        raise TypeError(f"Expected ray.data.Dataset, got {type(ray_dataset)}. "
                       f"Use run_standardscaler() wrapper for pandas input.")
    
    input_cols = spec.params.get("input_cols", spec.input_cols)
    output_cols = spec.params.get("output_cols", spec.output_cols)
    
    if _logger:
        _logger.info(f"StandardScaler: {input_cols} -> {output_cols}")
    
    # 遵循 Spark MLlib 标准行为：不保留原始列（最小 overhead）
    # 直接在 input_cols 上操作（原地替换）
    preprocessor = RayStandardScaler(columns=input_cols)
    fitted = preprocessor.fit(ray_dataset)
    result = fitted.transform(ray_dataset)
    
    # 如果输出列名不同，只需重命名（仍然不保留原始列）
    if input_cols != output_cols:
        rename_map = dict(zip(input_cols, output_cols))
        result = result.map_batches(
            lambda batch: batch.rename(columns=rename_map),
            batch_format="pandas"
        )
    
    return result


def run_standardscaler(input_data, spec: OperatorSpec):
    """
    便利wrapper - 支持pandas或Ray Dataset输入
    
    注意：benchmark代码应该直接调用run_standardscaler_with_ray_data
    """
    if isinstance(input_data, ray.data.Dataset):
        return run_standardscaler_with_ray_data(input_data, spec)
    elif isinstance(input_data, pd.DataFrame):
        ray_dataset = ray.data.from_pandas(input_data)
        return run_standardscaler_with_ray_data(ray_dataset, spec)
    else:
        raise TypeError(f"Expected ray.data.Dataset or pd.DataFrame, got {type(input_data)}")

