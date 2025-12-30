"""
Ray MinMaxScaler算子实现

完全基于Ray Data原生优化机制，无sklearn依赖。
支持分布式处理，使用map_batches进行向量化缩放。
"""

import pandas as pd
import ray.data
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_minmaxscaler(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    MinMaxScaler - pandas实现
    
    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 缩放后的DataFrame
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        min_val = spec.params.get("min", 0.0)
        max_val = spec.params.get("max", 1.0)

        result_df = input_df.copy()
        
        # 进行Min-Max缩放：(x - min) / (max - min) * (max_val - min_val) + min_val
        for input_col, output_col in zip(input_cols, output_cols):
            col_min = result_df[input_col].min()
            col_max = result_df[input_col].max()
            range_val = col_max - col_min
            result_df[output_col] = (result_df[input_col] - col_min) / (range_val + 1e-8) * (max_val - min_val) + min_val
        
        return result_df

    except Exception as e:
        raise RuntimeError(f"MinMaxScaler执行失败: {e}")


def run_minmaxscaler_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    MinMaxScaler - Ray Data实现
    
    使用map_batches进行分布式缩放，纯Ray Data，无sklearn依赖。
    
    Args:
        ray_dataset: Ray Dataset或pandas DataFrame
        spec: 算子规格

    Returns:
        Ray Dataset或DataFrame: 处理后的数据集
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        min_val = spec.params.get("min", 0.0)
        max_val = spec.params.get("max", 1.0)
        
        if _logger:
            _logger.info(f"MinMaxScaler: {input_cols} -> {output_cols}")

        # 处理pandas DataFrame输入
        if isinstance(ray_dataset, pd.DataFrame):
            return run_minmaxscaler(ray_dataset, spec)
        
        # Ray Dataset处理：使用map_batches进行向量化缩放
        def scale_fn(batch):
            """使用向量化操作进行MinMax缩放"""
            result = batch.copy()
            
            for input_col, output_col in zip(input_cols, output_cols):
                col_data = batch[input_col].astype('float64')
                col_min = col_data.min()
                col_max = col_data.max()
                range_val = col_max - col_min
                result[output_col] = (col_data - col_min) / (range_val + 1e-8) * (max_val - min_val) + min_val
            
            return result
        
        # 应用map_batches
        result = ray_dataset.map_batches(scale_fn, batch_format="pandas", batch_size=1024)
        return result

    except Exception as e:
        raise RuntimeError(f"Ray Data MinMaxScaler执行失败: {e}")


def run_minmaxscaler_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """简化的MinMaxScaler实现（使用Ray Data）"""
    return run_minmaxscaler(input_df, spec)
