"""
Spark Imputer算子实现
"""

from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer
from typing import List
from bench.operator_spec import OperatorSpec

try:
    from bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_imputer(spark,
                input_df: DataFrame,
                spec: OperatorSpec) -> DataFrame:
    """
    运行Imputer算子

    Args:
        spark: Spark会话（为了一致性接口，实际未使用）
        input_df: 输入Spark DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含填充后的列
    """
    try:
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        strategy = spec.params.get("strategy", "mean")
        # 注意：PySpark Imputer 不支持自定义fill_value，使用默认值

        if _logger:
            _logger.debug(f"Imputer参数: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"strategy={strategy}")
        else:
            print(f"DEBUG: Imputer参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: strategy={strategy}")

        # 检查输入列是否存在
        existing_cols = input_df.columns
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 初始化结果DataFrame
        result_df = input_df

        # 对每一列单独应用Imputer
        for input_col, output_col in zip(input_cols, output_cols):
            # 创建Imputer实例
            # 注意：PySpark Imputer的constant策略使用默认填充值（通常是0.0）
            imputer = Imputer(
                inputCol=input_col,
                outputCol=output_col,
                strategy=strategy
            )

            # 拟合和转换
            imputer_model = imputer.fit(result_df)
            result_df = imputer_model.transform(result_df)

        if _logger:
            _logger.info(f"Imputer处理完成，输出列: {result_df.columns}")
        else:
            print(f"Imputer处理完成，输出列: {result_df.columns}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return result_df

    except Exception as e:
        if _logger:
            _logger.error(f"Imputer执行失败: {e}", exc_info=True)
        else:
            print(f"ERROR: Imputer执行失败: {e}")
        raise RuntimeError(f"Imputer执行失败: {e}")
