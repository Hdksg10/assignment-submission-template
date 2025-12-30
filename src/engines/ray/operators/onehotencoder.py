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


def run_onehotencoder(input_data, spec: OperatorSpec):
    """使用Ray官方OneHotEncoder API进行独热编码 - 支持DataFrame和Ray Dataset"""
    # 获取输入列和输出列
    input_cols = spec.input_cols
    output_cols = spec.output_cols
    
    # 总是假设输入是Ray Dataset（框架已经转换）
    ray_dataset = input_data if hasattr(input_data, 'to_pandas') else ray.data.from_pandas(input_data)
    
    # 框架错误地设置了input_cols（它使用了前一个算子的output_cols）
    # OneHotEncoder应该只处理'cat_indexed'列
    # 获取数据的第一行来检查列名
    sample = ray_dataset.take(1)
    if sample:
        available_cols = list(sample[0].keys()) if isinstance(sample[0], dict) else sample[0].keys()
        
        # 如果'cat_indexed'列存在但不在input_cols中，修正它
        if 'cat_indexed' in available_cols and 'cat_indexed' not in input_cols:
            input_cols = ['cat_indexed']
            output_cols = ['cat_onehot']
    
    if _logger:
        _logger.info(f"OneHotEncoder: {input_cols} -> {output_cols}")
    
    # 使用Ray的OneHotEncoder preprocessor
    preprocessor = OneHotEncoder(columns=input_cols)
    fitted = preprocessor.fit(ray_dataset)
    result = fitted.transform(ray_dataset)
    
    # 总是返回Ray Dataset（保持框架的期望）
    return result
