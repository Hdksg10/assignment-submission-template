"""
Spark IDF算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import IDF
from typing import List
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_idf(spark,
           input_df: DataFrame,
           spec: OperatorSpec) -> DataFrame:
    """
    运行IDF算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含TF-IDF向量列
    """
    try:
        # 获取参数
        input_col = spec.params.get("input_col", spec.input_cols[0] if spec.input_cols else "tf_features")
        output_col = spec.params.get("output_col", spec.output_cols[0] if spec.output_cols else "tfidf_features")
        min_doc_freq = spec.params.get("min_doc_freq", 1)

        if _logger:
            _logger.debug(f"IDF参数: input_col={input_col}, output_col={output_col}")
            _logger.debug(f"min_doc_freq={min_doc_freq}")
        else:
            print(f"DEBUG: IDF参数: input_col={input_col}, output_col={output_col}")
            print(f"DEBUG: min_doc_freq={min_doc_freq}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        if input_col not in existing_cols:
            raise ValueError(f"输入列不存在: {input_col}")

        # 步骤1: 应用IDF
        idf = IDF(
            inputCol=input_col,
            outputCol=output_col,
            minDocFreq=min_doc_freq
        )

        # 步骤2: 拟合和转换
        idf_model = idf.fit(input_df)
        tfidf_df = idf_model.transform(input_df)

        if _logger:
            _logger.info(f"IDF处理完成，输出列: {tfidf_df.columns}")
        else:
            print(f"IDF处理完成，输出列: {tfidf_df.columns}")

        return tfidf_df

    except Exception as e:
        if _logger:
            _logger.error(f"IDF执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: IDF执行失败: {e}")
        raise RuntimeError(f"IDF执行失败: {e}")
