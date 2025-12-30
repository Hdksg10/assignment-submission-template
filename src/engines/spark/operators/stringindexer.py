"""
Spark StringIndexer算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_stringindexer(spark,
                      input_df: DataFrame,
                      spec: OperatorSpec) -> DataFrame:
    """
    运行StringIndexer算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含索引后的列
    """
    try:
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"StringIndexer参数: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"handle_invalid={handle_invalid}")
        else:
            print(f"DEBUG: StringIndexer参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: handle_invalid={handle_invalid}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量必须与输入列数量一致")

        # 转换handle_invalid参数为Spark格式
        # Spark StringIndexer支持: "error", "skip", "keep"
        spark_handle_invalid = handle_invalid
        if handle_invalid not in ["error", "skip", "keep"]:
            raise ValueError(f"不支持的handle_invalid策略: {handle_invalid}，支持: error, skip, keep")

        # 为每个输入列创建StringIndexer并应用
        result_df = input_df
        for input_col, output_col in zip(input_cols, output_cols):
            # 创建StringIndexer
            indexer = StringIndexer(
                inputCol=input_col,
                outputCol=output_col,
                handleInvalid=spark_handle_invalid
            )

            # 拟合和转换
            indexer_model = indexer.fit(result_df)
            result_df = indexer_model.transform(result_df)
            
            # 将输出列转换为整数类型（Spark StringIndexer默认输出double）
            result_df = result_df.withColumn(
                output_col,
                col(output_col).cast(IntegerType())
            )

        # StringIndexer 的特殊行为：保留原始列（因为需要验证映射关系）
        # Spark MLlib 的 StringIndexer 默认保留原始列（输出到新列）
        output_columns = existing_cols + output_cols
        final_df = result_df.select(*output_columns)

        if _logger:
            _logger.info(f"StringIndexer处理完成，输出列: {final_df.columns}")
        else:
            print(f"StringIndexer处理完成，输出列: {final_df.columns}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return final_df

    except Exception as e:
        raise RuntimeError(f"StringIndexer执行失败: {e}")

