"""
Spark StandardScaler算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import StandardScaler, VectorAssembler
from typing import List
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_standardscaler(spark,
                      input_df: DataFrame,
                      spec: OperatorSpec) -> DataFrame:
    """
    运行StandardScaler算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含标准化后的列
    """
    try:
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        with_mean = spec.params.get("with_mean", True)
        with_std = spec.params.get("with_std", True)

        if _logger:
            _logger.debug(f"StandardScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"with_mean={with_mean}, with_std={with_std}")
        else:
            print(f"DEBUG: StandardScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: with_mean={with_mean}, with_std={with_std}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 步骤1: 将数值列组装成向量
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="_vector_features"
        )
        assembled_df = assembler.transform(input_df)

        # 步骤2: 应用StandardScaler
        scaler = StandardScaler(
            inputCol="_vector_features",
            outputCol="_scaled_features",
            withMean=with_mean,
            withStd=with_std
        )

        # 拟合和转换
        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df)

        # 步骤3: 将缩放后的向量拆分回单独的列
        from pyspark.sql.functions import col

        # 提取缩放后的向量元素
        for i, (input_col, output_col) in enumerate(zip(input_cols, output_cols)):
            scaled_df = scaled_df.withColumn(
                output_col,
                col("_scaled_features")[i]
            )

        # 步骤4: 选择输出列（保留原始列和新的缩放列）
        output_columns = existing_cols + output_cols
        final_df = scaled_df.select(*output_columns)

        if _logger:
            _logger.info(f"StandardScaler处理完成，输出列: {final_df.columns}")
        else:
            print(f"StandardScaler处理完成，输出列: {final_df.columns}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return final_df

    except Exception as e:
        if _logger:
            _logger.error(f"StandardScaler执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: StandardScaler执行失败: {e}")
        raise RuntimeError(f"StandardScaler执行失败: {e}")


def run_standardscaler_pandas(spark,
                             pandas_df,
                             spec: OperatorSpec):
    """
    基于pandas的StandardScaler实现（用于对比）

    Args:
        spark: Spark会话
        pandas_df: pandas DataFrame
        spec: 算子规格

    Returns:
        pandas DataFrame: 处理后的DataFrame
    """
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        with_mean = spec.params.get("with_mean", True)
        with_std = spec.params.get("with_std", True)

        # 复制DataFrame
        result_df = pandas_df.copy()

        # 应用StandardScaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        scaled_data = scaler.fit_transform(result_df[input_cols])

        # 添加缩放后的列
        for i, col in enumerate(output_cols):
            result_df[col] = scaled_data[:, i]

        return result_df

    except Exception as e:
        if _logger:
            _logger.error(f"pandas StandardScaler执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: pandas StandardScaler执行失败: {e}")
        raise RuntimeError(f"pandas StandardScaler执行失败: {e}")
