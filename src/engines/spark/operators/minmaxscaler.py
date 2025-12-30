"""
Spark MinMaxScaler 算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # logger 不可用时的回退
    _logger = None


def run_minmaxscaler(spark,
                    input_df: DataFrame,
                    spec: OperatorSpec) -> DataFrame:
    """
    运行 MinMaxScaler 算子（Spark 版本）

    Args:
        spark: Spark 会话（保持签名一致，实际未直接使用）
        input_df: 输入 Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 包含缩放列的输出 DataFrame
    """
    try:
        # 参数提取
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        min_val = spec.params.get("min", 0.0)
        max_val = spec.params.get("max", 1.0)

        if _logger:
            _logger.debug(
                "MinMaxScaler参数: input_cols=%s, output_cols=%s, min=%s, max=%s",
                input_cols, output_cols, min_val, max_val
            )
        else:
            print(f"DEBUG: MinMaxScaler参数: input_cols={input_cols}, output_cols={output_cols}, min={min_val}, max={max_val}")

        # 输入列校验
        existing_cols = input_df.columns
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量必须与输入列数量一致")

        # 组装特征向量
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="_vector_features"
        )
        assembled_df = assembler.transform(input_df)

        # 进行 MinMax 缩放
        scaler = MinMaxScaler(
            inputCol="_vector_features",
            outputCol="_scaled_features",
            min=min_val,
            max=max_val
        )
        model = scaler.fit(assembled_df)
        scaled_df = model.transform(assembled_df)

        # 将向量转换为数组，然后提取各个元素
        # 使用 vector_to_array 处理稀疏向量和密集向量
        scaled_df = scaled_df.withColumn(
            "_scaled_array",
            vector_to_array(col("_scaled_features"))
        )
        
        # 从数组中提取各个元素
        for idx, out_col in enumerate(output_cols):
            scaled_df = scaled_df.withColumn(
                out_col,
                col("_scaled_array")[idx]
            )

        # 选择输出列：保留原始列并追加新列
        final_df = scaled_df.select(*(existing_cols + output_cols))

        if _logger:
            _logger.info("MinMaxScaler处理完成，输出列: %s", final_df.columns)
        else:
            print(f"MinMaxScaler处理完成，输出列: {final_df.columns}")

        return final_df

    except Exception as exc:
        raise RuntimeError(f"MinMaxScaler执行失败: {exc}")

