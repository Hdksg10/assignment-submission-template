"""
Ray MinMaxScaler算子 - 使用Ray Data官方preprocessor API
"""
import pandas as pd
import ray.data
from ray.data.preprocessors import MinMaxScaler as RayMinMaxScaler
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_minmaxscaler(input_data, spec: OperatorSpec):
    """使用Ray官方MinMaxScaler API（缩放到[0,1]）- 支持DataFrame和Ray Dataset"""
    # 使用 spec 的 input_cols 和 output_cols（框架可能已修改）
    input_cols = spec.input_cols
    output_cols = spec.output_cols
    
    if _logger:
        _logger.info(f"MinMaxScaler: {input_cols} -> {output_cols}")
    
    # 总是假设输入是Ray Dataset（框架已经转换）
    ray_dataset = input_data if hasattr(input_data, 'to_pandas') else ray.data.from_pandas(input_data)
    
    # 使用Ray的MinMaxScaler preprocessor
    preprocessor = RayMinMaxScaler(columns=input_cols)
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
