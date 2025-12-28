"""
Ray StandardScaler算子实现

遵循设计原则：Stateful算子必须拆成 fit → transform 模式。

实现路径优先级（按设计原则）：
1. 优先：Ray Data preprocessor（ray.data.preprocessors.StandardScaler）
   - 当 with_mean=True 且 with_std=True 时使用
   - 利用 Ray Data 原生的优化机制
2. 其次：自研 fit/transform（当需要支持 with_mean/with_std 开关时）
   - 先全局聚合统计量（mean/std），再使用这些统计量进行transform
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


def run_standardscaler(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    兼容性函数：直接处理pandas DataFrame

    主要用于测试和兼容性，生产环境建议使用run_standardscaler_with_ray_data
    此函数使用scikit-learn的StandardScaler，确保与fit/transform模式语义一致。
    
    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含标准化后的列（dtype=float64）
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        with_mean = spec.params.get("with_mean", True)
        with_std = spec.params.get("with_std", True)

        if _logger:
            _logger.debug(f"Ray StandardScaler (pandas兼容模式): input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"with_mean={with_mean}, with_std={with_std}")
        else:
            print(f"DEBUG: Ray StandardScaler参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: with_mean={with_mean}, with_std={with_std}")

        # 检查输入列是否存在
        existing_cols = input_df.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = input_df.copy()

        # 创建StandardScaler并fit
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # 选择输入列数据，确保为float64
        input_data = result_df[input_cols].values.astype(np.float64)

        # 应用标准化（fit + transform）
        scaled_data = scaler.fit_transform(input_data)

        # 将结果添加到输出列，确保dtype为float64
        for i, output_col in enumerate(output_cols):
            result_df[output_col] = scaled_data[:, i].astype(np.float64)

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


def _fit_standardscaler(ray_dataset, input_cols: list, with_mean: bool, with_std: bool) -> Dict[str, np.ndarray]:
    """
    Fit阶段：全局聚合统计量（mean和std）
    
    遵循设计原则：Stateful算子必须先全局fit，禁止在map_batches内fit。
    
    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        with_mean: 是否计算均值
        with_std: 是否计算标准差
        
    Returns:
        Dict包含 'mean' 和 'std' 数组（如果对应开关为True）
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
            'sum': np.sum(input_data, axis=0) if with_mean else None,
            'sum_sq': np.sum(input_data ** 2, axis=0) if with_std else None,
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
    
    # 计算全局均值
    if with_mean:
        total_sum = np.sum([s['sum'] for s in batch_stats], axis=0)
        global_mean = total_sum / total_count
    else:
        global_mean = None
    
    # 计算全局标准差
    if with_std:
        if with_mean:
            # 使用两遍扫描方法：先算均值，再算方差（确保数值精度）
            def compute_variance_batch(batch: pd.DataFrame) -> Dict[str, Any]:
                input_data = batch[input_cols].values.astype(np.float64)
                centered = input_data - global_mean
                return {
                    'sum_sq_diff': np.sum(centered ** 2, axis=0),
                    'count': len(input_data)
                }
            
            variance_stats = ray_dataset.map_batches(
                compute_variance_batch,
                batch_format="pandas",
                batch_size=4096,
                compute="actors"
            ).take_all()
            
            total_sum_sq_diff = np.sum([s['sum_sq_diff'] for s in variance_stats], axis=0)
            variance = total_sum_sq_diff / total_count
        else:
            # 如果不需要均值，计算未中心化的标准差
            # std = sqrt(mean(x^2))
            total_sum_sq = np.sum([s['sum_sq'] for s in batch_stats], axis=0)
            mean_sq = total_sum_sq / total_count
            variance = mean_sq  # 未中心化的方差
        
        # 处理常数列（std=0的情况，避免除零）
        global_std = np.sqrt(np.maximum(variance, 0))  # 确保非负
        global_std = np.where(global_std == 0, 1.0, global_std)  # 常数列设为1.0，避免除零
    else:
        global_std = None
    
    return {
        'mean': global_mean,
        'std': global_std
    }


def _transform_standardscaler(ray_dataset, input_cols: list, output_cols: list, 
                              stats: Dict[str, np.ndarray], with_mean: bool, with_std: bool):
    """
    Transform阶段：使用全局统计量进行标准化
    
    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        output_cols: 输出列名列表
        stats: fit阶段得到的统计量（包含mean和std）
        with_mean: 是否应用均值中心化
        with_std: 是否应用标准差缩放
        
    Returns:
        Ray Dataset: 处理后的数据集
    """
    mean = stats.get('mean')
    std = stats.get('std')
    
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
        
        # 应用标准化变换
        if with_mean and mean is not None:
            input_data = input_data - mean
        if with_std and std is not None:
            input_data = input_data / std
        
        # 将结果添加到输出列，确保dtype为float64
        for i, output_col in enumerate(output_cols):
            result_df[output_col] = input_data[:, i].astype(np.float64)
        
        return result_df
    
    # 应用转换
    transformed_dataset = ray_dataset.map_batches(
        transform_batch,
        batch_format="pandas",  # 固定batch_format（遵循工程约束）
        batch_size=4096,
        compute="actors"
    )
    
    return transformed_dataset


def run_standardscaler_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray Data实现StandardScaler - 遵循fit/transform模式
    
    实现路径优先级（按设计原则）：
    1. 优先：Ray Data preprocessor（若支持所需语义/参数）
    2. 其次：自研 fit/transform（当需要支持 with_mean/with_std 开关时）
    
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
        with_mean = spec.params.get("with_mean", True)
        with_std = spec.params.get("with_std", True)
        
        if _logger:
            _logger.debug(f"Ray StandardScaler: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"with_mean={with_mean}, with_std={with_std}")

        # 检查是否可以使用 Ray Data preprocessor
        # Ray Data preprocessor 只支持默认的 with_mean=True, with_std=True
        use_ray_preprocessor = (with_mean is True and with_std is True)
        
        if use_ray_preprocessor:
            # 优先使用 Ray Data 内置 preprocessor
            try:
                from ray.data.preprocessors import StandardScaler as RayStandardScaler
                
                if _logger:
                    _logger.debug("使用 Ray Data preprocessor (StandardScaler)")
                
                # 创建并fit preprocessor
                preprocessor = RayStandardScaler(
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
        
        # 使用自研 fit/transform 实现（支持 with_mean/with_std 开关）
        if _logger:
            _logger.debug("使用自研 fit/transform 实现（支持 with_mean/with_std 开关）")
        
        # Step 1: Fit阶段 - 全局聚合统计量
        stats = _fit_standardscaler(ray_dataset, input_cols, with_mean, with_std)
        
        if _logger:
            _logger.debug(f"Fit完成: mean={stats.get('mean')}, std={stats.get('std')}")
        
        # Step 2: Transform阶段 - 使用全局统计量进行标准化
        transformed_dataset = _transform_standardscaler(
            ray_dataset, input_cols, output_cols, stats, with_mean, with_std
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
