"""
Spark Tokenizer算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from typing import List
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_tokenizer(spark,
                  input_df: DataFrame,
                  spec: OperatorSpec) -> DataFrame:
    """
    运行Tokenizer算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含分词后的列
    """
    try:
        # 获取参数
        input_col = spec.params.get("input_col", spec.input_cols[0] if spec.input_cols else "text")
        output_col = spec.params.get("output_col", spec.output_cols[0] if spec.output_cols else "tokens")
        pattern = spec.params.get("pattern", None)  # 如果提供pattern，使用RegexTokenizer

        if _logger:
            _logger.debug(f"Tokenizer参数: input_col={input_col}, output_col={output_col}")
            _logger.debug(f"pattern={pattern}")
        else:
            print(f"DEBUG: Tokenizer参数: input_col={input_col}, output_col={output_col}")
            print(f"DEBUG: pattern={pattern}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        if input_col not in existing_cols:
            raise ValueError(f"输入列不存在: {input_col}")

        # 步骤1: 选择使用Tokenizer还是RegexTokenizer
        if pattern is not None:
            # 使用RegexTokenizer进行自定义分词
            tokenizer = RegexTokenizer(
                inputCol=input_col,
                outputCol=output_col,
                pattern=pattern
            )
        else:
            # 使用默认Tokenizer（按空格分词）
            tokenizer = Tokenizer(
                inputCol=input_col,
                outputCol=output_col
            )

        # 步骤2: 应用分词器
        tokenized_df = tokenizer.transform(input_df)

        if _logger:
            _logger.info(f"Tokenizer处理完成，输出列: {tokenized_df.columns}")
        else:
            print(f"Tokenizer处理完成，输出列: {tokenized_df.columns}")

        return tokenized_df

    except Exception as e:
        if _logger:
            _logger.error(f"Tokenizer执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: Tokenizer执行失败: {e}")
        raise RuntimeError(f"Tokenizer执行失败: {e}")
