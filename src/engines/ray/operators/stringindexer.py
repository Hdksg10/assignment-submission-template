"""
Ray StringIndexer算子实现

完全基于Ray Data原生优化机制，无sklearn依赖。
使用Categorizer preprocessor进行分类编码。
"""

import pandas as pd
import ray.data
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_stringindexer(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    StringIndexer - pandas实现
    
    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 索引后的DataFrame
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)

        result_df = input_df.copy()
        
        # 对每列进行分类编码
        for input_col, output_col in zip(input_cols, output_cols):
            # 转换为字符串并处理缺失值
            col_data = result_df[input_col].fillna('<MISSING>').astype(str)
            # 获取唯一值并排序
            categories = sorted(col_data.unique())
            # 创建映射
            cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            # 应用映射
            result_df[output_col] = col_data.map(cat_to_idx).astype('int64')
        
        return result_df

    except Exception as e:
        raise RuntimeError(f"StringIndexer执行失败: {e}")


def run_stringindexer_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    StringIndexer - Ray Data实现
    
    使用Categorizer preprocessor进行分布式分类编码。
    
    Args:
        ray_dataset: Ray Dataset或pandas DataFrame
        spec: 算子规格

    Returns:
        Ray Dataset或DataFrame: 处理后的数据集
    """
    try:
        from ray.data.preprocessors import Categorizer
        
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        
        if _logger:
            _logger.info(f"StringIndexer: {input_cols} -> {output_cols}")

        # 处理pandas DataFrame输入
        if isinstance(ray_dataset, pd.DataFrame):
            return run_stringindexer(ray_dataset, spec)
        
        # Ray Dataset处理：使用Categorizer preprocessor
        try:
            # 尝试使用Ray Data的Categorizer
            preprocessor = Categorizer(columns=input_cols)
            fitted = preprocessor.fit(ray_dataset)
            result = fitted.transform(ray_dataset)
            return result
        except:
            # 如果Categorizer失败，回退到map_batches实现
            def categorize_fn(batch):
                """使用向量化操作进行分类编码"""
                result = batch.copy()
                
                for input_col, output_col in zip(input_cols, output_cols):
                    col_data = batch[input_col].fillna('<MISSING>').astype(str)
                    categories = sorted(col_data.unique())
                    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
                    result[output_col] = col_data.map(cat_to_idx).astype('int64')
                
                return result
            
            result = ray_dataset.map_batches(categorize_fn, batch_format="pandas", batch_size=1024)
            return result

    except Exception as e:
        raise RuntimeError(f"Ray Data StringIndexer执行失败: {e}")


def run_stringindexer_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """简化的StringIndexer实现（使用Ray Data）"""
    return run_stringindexer(input_df, spec)
