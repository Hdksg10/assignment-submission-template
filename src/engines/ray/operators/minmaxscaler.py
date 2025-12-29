"""
Ray MinMaxScaler算子实现

遵循设计原则：Stateful算子必须拆成 fit → transform 模式。

实现路径优先级（按设计原则）：
1. 优先：Ray Data preprocessor（ray.data.preprocessors.MinMaxScaler）
   - 当 min=0.0 且 max=1.0 时使用
   - 利用 Ray Data 原生的优化机制
2. 其次：自研 fit/transform（当需要支持任意 min/max 范围时）
   - 先全局聚合统计量（min/max），再使用这些统计量进行transform
   - 支持完整的参数对齐（Ray-Extended策略）
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_minmaxscaler(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    兼容性函数：直接处理pandas DataFrame

    主要用于测试和兼容性，生产环境建议使用run_minmaxscaler_with_ray_data
    此函数使用scikit-learn的MinMaxScaler，确保与fit/transform模式语义一致。

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含缩放后的列（dtype=float64）
    """
    try:
        from sklearn.preprocessing import MinMaxScaler

        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        min_val = spec.params.get("min", 0.0)
        max_val = spec.params.get("max", 1.0)

        if _logger:
            _logger.debug(f"Ray MinMaxScaler (pandas兼容模式): input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"min={min_val}, max={max_val}")
        else:
            print(f"DEBUG: Ray MinMaxScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: min={min_val}, max={max_val}")

        # 检查输入列是否存在
        existing_cols = input_df.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = input_df.copy()

        # 创建MinMaxScaler并fit
        scaler = MinMaxScaler(feature_range=(min_val, max_val))

        # 选择输入列数据，确保为float64
        input_data = result_df[input_cols].values.astype(np.float64)

        # 应用缩放（fit + transform）
        scaled_data = scaler.fit_transform(input_data)

        # 将结果添加到输出列，确保dtype为float64
        for i, output_col in enumerate(output_cols):
            result_df[output_col] = scaled_data[:, i].astype(np.float64)

        if _logger:
            _logger.info(f"Ray MinMaxScaler处理完成，输出列: {result_df.columns.tolist()}")
        else:
            print(f"Ray MinMaxScaler处理完成，输出列: {result_df.columns.tolist()}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Ray MinMaxScaler执行失败: {e}")


def _fit_minmaxscaler(ray_dataset, input_cols: list, min_val: float, max_val: float) -> Dict[str, np.ndarray]:
    """
    Fit阶段：全局聚合统计量（min和max）

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        min_val: 目标最小值
        max_val: 目标最大值

    Returns:
        Dict包含 'min_vals' 和 'max_vals' 数组
    """

    def compute_batch_stats(batch: pd.DataFrame) -> Dict[str, Any]:
        """计算单个batch的统计量"""
        # 检查输入列是否存在
        existing_cols = batch.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 选择输入列数据，确保为float64
        input_data = batch[input_cols].values.astype(np.float64)

        # 计算统计量
        stats = {
            'count': len(input_data),
            'min_vals': np.min(input_data, axis=0),
            'max_vals': np.max(input_data, axis=0),
        }

        return stats

    # 收集所有batch的统计量
    batch_stats = ray_dataset.map_batches(
        compute_batch_stats,
        batch_format="pandas",
        batch_size=4096,
        compute="actors"
    ).take_all()

    # 全局聚合统计量
    total_count = sum(s['count'] for s in batch_stats)

    if total_count == 0:
        raise ValueError("数据集为空，无法计算统计量")

    # 计算全局min和max
    all_min_vals = np.array([s['min_vals'] for s in batch_stats])
    all_max_vals = np.array([s['max_vals'] for s in batch_stats])

    global_min_vals = np.min(all_min_vals, axis=0)
    global_max_vals = np.max(all_max_vals, axis=0)

    return {
        'min_vals': global_min_vals,
        'max_vals': global_max_vals
    }


def _transform_minmaxscaler(ray_dataset, input_cols: list, output_cols: list,
                           stats: Dict[str, np.ndarray], min_val: float, max_val: float):
    """
    Transform阶段：使用全局统计量进行缩放

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        output_cols: 输出列名列表
        stats: fit阶段得到的统计量（包含min_vals和max_vals）
        min_val: 目标最小值
        max_val: 目标最大值

    Returns:
        Ray Dataset: 处理后的数据集
    """
    min_vals = stats.get('min_vals')
    max_vals = stats.get('max_vals')

    def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """批处理转换函数 - 使用全局统计量"""
        # 检查输入列是否存在
        existing_cols = batch.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = batch.copy()

        # 选择输入列数据，确保为float64
        input_data = result_df[input_cols].values.astype(np.float64)

        # 应用MinMax缩放变换
        # 处理常数列（max_vals == min_vals的情况，避免除零）
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1.0, ranges)  # 常数列设为1.0

        # 标准化到[0,1]范围
        scaled_data = (input_data - min_vals) / ranges

        # 缩放到目标范围[min_val, max_val]
        scaled_data = scaled_data * (max_val - min_val) + min_val

        # 将结果添加到输出列，确保dtype为float64
        for i, output_col in enumerate(output_cols):
            result_df[output_col] = scaled_data[:, i].astype(np.float64)

        return result_df

    # 应用转换
    transformed_dataset = ray_dataset.map_batches(
        transform_batch,
        batch_format="pandas",  # 固定batch_format（遵循工程约束）
        batch_size=4096,
        compute="actors"
    )

    return transformed_dataset


def run_minmaxscaler_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray Data实现MinMaxScaler - 遵循fit/transform模式

    实现路径优先级（按设计原则）：
    1. 优先：Ray Data preprocessor（若能满足对齐语义/参数要求）
    2. 其次：自研 fit/transform（当需要支持任意 min/max 范围时）

    核心设计原则：
    1. Stateful算子必须先全局fit，再transform
    2. 禁止在map_batches内fit（避免每个batch各自fit的错误语义）
    3. 输出dtype统一为float64
    4. 不就地修改输入batch

    Args:
        ray_dataset: Ray Dataset
        spec: 算子规格

    Returns:
        Ray Dataset: 处理后的数据集
    """
    try:
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        min_val = spec.params.get("min", 0.0)
        max_val = spec.params.get("max", 1.0)

        if _logger:
            _logger.debug(f"Ray MinMaxScaler: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"min={min_val}, max={max_val}")

        # 检查是否可以使用 Ray Data preprocessor
        # Ray Data preprocessor 只支持默认的 [0,1] 范围
        use_ray_preprocessor = (min_val == 0.0 and max_val == 1.0)

        if use_ray_preprocessor:
            # 优先使用 Ray Data 内置 preprocessor
            try:
                from ray.data.preprocessors import MinMaxScaler as RayMinMaxScaler

                if _logger:
                    _logger.debug("使用 Ray Data preprocessor (MinMaxScaler)")

                # 创建并fit preprocessor
                preprocessor = RayMinMaxScaler(
                    columns=input_cols,
                    output_columns=output_cols
                )
                fitted_preprocessor = preprocessor.fit(ray_dataset)

                # Transform
                transformed_dataset = fitted_preprocessor.transform(ray_dataset)

                # 确保输出dtype为float64（Ray preprocessor可能不保证）
                def ensure_float64(batch: pd.DataFrame) -> pd.DataFrame:
                    result_df = batch.copy()
                    for col in output_cols:
                        if col in result_df.columns:
                            result_df[col] = result_df[col].astype(np.float64)
                    return result_df

                transformed_dataset = transformed_dataset.map_batches(
                    ensure_float64,
                    batch_format="pandas",
                    batch_size=4096
                )

                if _logger:
                    _logger.debug("Ray Data preprocessor 执行完成")

                return transformed_dataset

            except Exception as e:
                # 如果 Ray preprocessor 失败，回退到自研实现
                if _logger:
                    _logger.warning(f"Ray Data preprocessor 失败，回退到自研实现: {e}")
                # 继续执行自研实现

        # 使用自研 fit/transform 实现（支持任意 min/max 范围）
        if _logger:
            _logger.debug("使用自研 fit/transform 实现（支持任意 min/max 范围）")

        # Step 1: Fit阶段 - 全局聚合统计量
        stats = _fit_minmaxscaler(ray_dataset, input_cols, min_val, max_val)

        if _logger:
            _logger.debug(f"Fit完成: min_vals={stats.get('min_vals')}, max_vals={stats.get('max_vals')}")

        # Step 2: Transform阶段 - 使用全局统计量进行缩放
        transformed_dataset = _transform_minmaxscaler(
            ray_dataset, input_cols, output_cols, stats, min_val, max_val
        )

        return transformed_dataset

    except Exception as e:
        raise RuntimeError(f"Ray Data MinMaxScaler执行失败: {e}")


def run_minmaxscaler_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    简化的MinMaxScaler实现（直接使用scikit-learn）

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame
    """
    return run_minmaxscaler(input_df, spec)