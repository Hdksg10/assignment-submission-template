"""
Ray StandardScaler算子实现

统一使用Ray Data以最小化包装开销。
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_standardscaler(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    兼容性函数：直接处理pandas DataFrame

    主要用于测试和兼容性，生产环境建议使用run_standardscaler_with_ray_data
    """
    """
    运行StandardScaler算子（基于Ray Data）

    Args:
        input_df: 输入pandas DataFrame
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
            _logger.debug(f"Ray StandardScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"with_mean={with_mean}, with_std={with_std}")
        else:
            print(f"DEBUG: Ray StandardScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: with_mean={with_mean}, with_std={with_std}")

        # 检查输入列是否存在
        existing_cols = input_df.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据
        result_df = input_df.copy()

        # 创建StandardScaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

        # 选择输入列数据
        input_data = result_df[input_cols].values

        # 应用标准化
        scaled_data = scaler.fit_transform(input_data)

        # 将结果添加到输出列
        for i, output_col in enumerate(output_cols):
            result_df[output_col] = scaled_data[:, i]

        if _logger:
            _logger.info(f"Ray StandardScaler处理完成，输出列: {result_df.columns.tolist()}")
        else:
            print(f"Ray StandardScaler处理完成，输出列: {result_df.columns.tolist()}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Ray StandardScaler执行失败: {e}")


def run_standardscaler_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray Data的map_batches实现StandardScaler - 高性能版本

    Args:
        ray_dataset: Ray Dataset
        spec: 算子规格

    Returns:
        Ray Dataset: 处理后的数据集
    """
    try:
        import ray.data as rd

        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        with_mean = spec.params.get("with_mean", True)
        with_std = spec.params.get("with_std", True)

        def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """批处理转换函数 - 内联实现减少函数调用开销"""
            # 直接在batch上操作，避免额外的函数调用
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

            # 检查输入列是否存在
            existing_cols = batch.columns.tolist()
            missing_cols = [col for col in input_cols if col not in existing_cols]
            if missing_cols:
                raise ValueError(f"输入列不存在: {missing_cols}")

            # 复制DataFrame避免修改原数据
            result_df = batch.copy()

            # 选择输入列数据
            input_data = result_df[input_cols].values

            # 应用标准化
            scaled_data = scaler.fit_transform(input_data)

            # 将结果添加到输出列
            for i, output_col in enumerate(output_cols):
                result_df[output_col] = scaled_data[:, i]

            return result_df

        # 应用转换 - 使用更大的batch_size以减少overhead
        transformed_dataset = ray_dataset.map_batches(
            transform_batch,
            batch_format="pandas",
            batch_size=4096,  # 增大batch_size减少map_batches调用次数
            compute="actors"  # 使用actors进行计算以提高性能
        )

        return transformed_dataset

    except Exception as e:
        raise RuntimeError(f"Ray Data StandardScaler执行失败: {e}")


def run_standardscaler_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    简化的StandardScaler实现（直接使用scikit-learn）

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame
    """
    return run_standardscaler(input_df, spec)
