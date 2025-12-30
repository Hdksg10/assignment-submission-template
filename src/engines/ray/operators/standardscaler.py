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


def run_standardscaler(input_data, spec: OperatorSpec):
    """使用Ray官方StandardScaler API - 支持DataFrame和Ray Dataset"""
    input_cols = spec.params.get("input_cols", spec.input_cols)
    output_cols = spec.params.get("output_cols", spec.output_cols)
    
    if _logger:
        _logger.info(f"StandardScaler: {input_cols} -> {output_cols}")
    
    # 总是假设输入是Ray Dataset（框架已经转换）
    ray_dataset = input_data if hasattr(input_data, 'to_pandas') else ray.data.from_pandas(input_data)
    
    # 使用Ray的StandardScaler preprocessor
    preprocessor = RayStandardScaler(columns=input_cols)
    fitted = preprocessor.fit(ray_dataset)
    result = fitted.transform(ray_dataset)
    
    # 列重命名
    if input_cols != output_cols:
        rename_map = dict(zip(input_cols, output_cols))
        result = result.map_batches(
            lambda batch: batch.rename(columns=rename_map),
            batch_format="pandas"
        )
    
    # 总是返回Ray Dataset（保持框架的期望）
    return result

