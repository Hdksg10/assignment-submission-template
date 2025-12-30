"""
Spark OneHotEncoder算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import OneHotEncoder
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_onehotencoder(spark,
                     input_df: DataFrame,
                     spec: OperatorSpec) -> DataFrame:
    """
    运行OneHotEncoder算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含独热编码后的列
    """
    try:
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        drop_last = spec.params.get("drop_last", True)
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"OneHotEncoder参数: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"drop_last={drop_last}, handle_invalid={handle_invalid}")
        else:
            print(f"DEBUG: OneHotEncoder参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: drop_last={drop_last}, handle_invalid={handle_invalid}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        missing_cols = [col_name for col_name in input_cols if col_name not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量必须与输入列数量一致")

        # 转换handle_invalid参数为Spark格式
        # Spark OneHotEncoder支持: "error", "keep"
        spark_handle_invalid = handle_invalid
        if handle_invalid not in ["error", "keep"]:
            raise ValueError(f"不支持的handle_invalid策略: {handle_invalid}，支持: error, keep")

        # 为每个输入列创建OneHotEncoder并应用
        result_df = input_df
        for input_col, output_col in zip(input_cols, output_cols):
            # 创建OneHotEncoder
            # Spark OneHotEncoder的dropLast参数对应drop_last
            encoder = OneHotEncoder(
                inputCol=input_col,
                outputCol=output_col,
                dropLast=drop_last,
                handleInvalid=spark_handle_invalid
            )

            # 拟合和转换
            encoder_model = encoder.fit(result_df)
            result_df = encoder_model.transform(result_df)

        # 选择输出列（遵循MLlib标准：不保留原始input_cols）
        keep_cols = [c for c in existing_cols if c not in input_cols] + output_cols
        final_df = result_df.select(*keep_cols)

        if _logger:
            _logger.info(f"OneHotEncoder处理完成，输出列: {final_df.columns}")
        else:
            print(f"OneHotEncoder处理完成，输出列: {final_df.columns}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return final_df

    except Exception as e:
        raise RuntimeError(f"OneHotEncoder执行失败: {e}")

