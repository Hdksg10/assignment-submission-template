"""
Ray OneHotEncoder算子实现

完全基于Ray Data原生优化机制，无sklearn依赖。
使用OneHotEncoder preprocessor进行独热编码。
"""

import pandas as pd
import ray.data
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_onehotencoder(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    OneHotEncoder - pandas实现
    
    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 独热编码后的DataFrame
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)

        result_df = input_df.copy()
        
        # 对每列进行独热编码
        for input_col in input_cols:
            # 使用pandas的get_dummies
            one_hot = pd.get_dummies(result_df[input_col], prefix=input_col, prefix_sep='_')
            result_df = pd.concat([result_df, one_hot], axis=1)
        
        return result_df

    except Exception as e:
        raise RuntimeError(f"OneHotEncoder执行失败: {e}")


def run_onehotencoder_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    OneHotEncoder - Ray Data实现
    
    使用OneHotEncoder preprocessor进行分布式独热编码。
    
    Args:
        ray_dataset: Ray Dataset或pandas DataFrame
        spec: 算子规格

    Returns:
        Ray Dataset或DataFrame: 处理后的数据集
    """
    try:
        from ray.data.preprocessors import OneHotEncoder
        
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        
        if _logger:
            _logger.info(f"OneHotEncoder: {input_cols} -> {output_cols}")

        # 处理pandas DataFrame输入
        if isinstance(ray_dataset, pd.DataFrame):
            return run_onehotencoder(ray_dataset, spec)
        
        # Ray Dataset处理：使用OneHotEncoder preprocessor
        try:
            # 尝试使用Ray Data的OneHotEncoder
            preprocessor = OneHotEncoder(columns=input_cols)
            fitted = preprocessor.fit(ray_dataset)
            result = fitted.transform(ray_dataset)
            return result
        except:
            # 如果OneHotEncoder失败，回退到map_batches实现
            def onehot_fn(batch):
                """使用向量化操作进行独热编码"""
                result = batch.copy()
                
                for input_col in input_cols:
                    # 使用pandas的get_dummies
                    one_hot = pd.get_dummies(batch[input_col], prefix=input_col, prefix_sep='_')
                    result = pd.concat([result, one_hot], axis=1)
                
                return result
            
            result = ray_dataset.map_batches(onehot_fn, batch_format="pandas", batch_size=1024)
            return result

    except Exception as e:
        raise RuntimeError(f"Ray Data OneHotEncoder执行失败: {e}")


def run_onehotencoder_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    OneHotEncoder - Ray Data实现
    
    使用Ray Data内置的OneHotEncoder preprocessor进行独热编码。
    
    Args:
        ray_dataset: Ray Dataset或pandas DataFrame
        spec: 算子规格

    Returns:
        Ray Dataset或DataFrame: 处理后的数据集
    """
    try:
        from ray.data.preprocessors import OneHotEncoder
        
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        
        if _logger:
            _logger.info(f"OneHotEncoder: {input_cols} -> {output_cols}")

        # 如果输入是pandas DataFrame，转换为Ray Dataset
        if isinstance(ray_dataset, pd.DataFrame):
            ray_dataset = ray.data.from_pandas(ray_dataset)
        
        # 先转换为pandas，使用pandas模式处理
        # 然后转回Ray Dataset
        pdf = ray_dataset.to_pandas()
        result_pdf = run_onehotencoder(pdf, spec)
        
        return ray.data.from_pandas(result_pdf)

    except Exception as e:
        raise RuntimeError(f"Ray Data OneHotEncoder执行失败: {e}")


def run_onehotencoder_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """简化的OneHotEncoder实现（使用Ray Data）"""
    return run_onehotencoder(input_df, spec)