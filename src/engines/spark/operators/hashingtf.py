"""
Spark HashingTF算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import HashingTF
from typing import List
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_hashingtf(spark,
                  input_df: DataFrame,
                  spec: OperatorSpec) -> DataFrame:
    """
    运行HashingTF算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含哈希向量化后的列
    """
    try:
        # 获取参数
        input_col = spec.params.get("input_col", spec.input_cols[0] if spec.input_cols else "tokens")
        output_col = spec.params.get("output_col", spec.output_cols[0] if spec.output_cols else "tf_features")
        num_features = spec.params.get("num_features", 2**18)  # 默认2^18 = 262144

        if _logger:
            _logger.debug(f"HashingTF参数: input_col={input_col}, output_col={output_col}")
            _logger.debug(f"num_features={num_features}")
        else:
            print(f"DEBUG: HashingTF参数: input_col={input_col}, output_col={output_col}")
            print(f"DEBUG: num_features={num_features}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        if input_col not in existing_cols:
            raise ValueError(f"输入列不存在: {input_col}")

        # 步骤1: 应用HashingTF
        hashing_tf = HashingTF(
            inputCol=input_col,
            outputCol=output_col,
            numFeatures=num_features
        )

        # 步骤2: 转换
        hashed_df = hashing_tf.transform(input_df)

        if _logger:
            _logger.info(f"HashingTF处理完成，输出列: {hashed_df.columns}")
        else:
            print(f"HashingTF处理完成，输出列: {hashed_df.columns}")

        return hashed_df

    except Exception as e:
        if _logger:
            _logger.error(f"HashingTF执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: HashingTF执行失败: {e}")
        raise RuntimeError(f"HashingTF执行失败: {e}")
