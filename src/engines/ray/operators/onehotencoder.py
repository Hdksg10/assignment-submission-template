"""
Ray OneHotEncoder算子实现

遵循设计原则：Stateful算子必须拆成 fit → transform 模式。

实现路径优先级（按设计原则）：
1. 优先：Ray Data preprocessor（若可用）
2. 其次：自研 fit/transform
   - 先全局统计类别数量（通常依赖StringIndexer的结果）
   - 再使用类别数量进行独热编码
   - 支持drop_last和handle_invalid策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def run_onehotencoder(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    兼容性函数：直接处理pandas DataFrame

    主要用于测试和兼容性，生产环境建议使用run_onehotencoder_with_ray_data
    此函数使用scikit-learn的OneHotEncoder，确保与fit/transform模式语义一致。

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含独热编码后的列（dtype=float64）
    """
    try:
        from sklearn.preprocessing import OneHotEncoder

        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        drop_last = spec.params.get("drop_last", True)
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"Ray OneHotEncoder (pandas兼容模式): input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"drop_last={drop_last}, handle_invalid={handle_invalid}")
        else:
            print(f"DEBUG: Ray OneHotEncoder参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: drop_last={drop_last}, handle_invalid={handle_invalid}")

        # 检查输入列是否存在
        existing_cols = input_df.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = input_df.copy()

        # 为每个输入列创建OneHotEncoder并应用
        for input_col, output_col in zip(input_cols, output_cols):
            # 获取列数据
            col_data = result_df[[input_col]]

            # 创建OneHotEncoder并fit
            encoder = OneHotEncoder(
                drop='first' if drop_last else None,
                handle_unknown='error' if handle_invalid == 'error' else 'ignore',
                sparse_output=False  # 现代sklearn版本使用sparse_output
            )

            encoded_data = encoder.fit_transform(col_data)

            # 为独热编码列创建列名
            feature_names = []
            categories = encoder.categories_[0]
            for i, category in enumerate(categories):
                if drop_last and i == 0:
                    continue  # 跳过第一个类别（drop='first'）
                feature_names.append(f"{output_col}_{category}")

            # 创建独热编码DataFrame
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=feature_names,
                index=result_df.index,
                dtype=np.float64
            )

            # 将独热编码列添加到结果DataFrame
            result_df = pd.concat([result_df, encoded_df], axis=1)

        if _logger:
            _logger.info(f"Ray OneHotEncoder处理完成，输出列: {result_df.columns.tolist()}")
        else:
            print(f"Ray OneHotEncoder处理完成，输出列: {result_df.columns.tolist()}")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Ray OneHotEncoder执行失败: {e}")


def _fit_onehotencoder(ray_dataset, input_cols: list) -> Dict[str, Any]:
    """
    Fit阶段：全局统计类别数量

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表

    Returns:
        Dict包含类别数量信息
    """

    def collect_category_counts(batch: pd.DataFrame) -> pd.DataFrame:
        """收集单个batch的类别计数并返回为DataFrame"""
        category_counts = {}
        for col in input_cols:
            if col in batch.columns:
                # 计算唯一值数量
                unique_count = batch[col].nunique()
                category_counts[col] = unique_count
        # 返回DataFrame而不是dict，以避免Ray Data的格式要求问题
        return pd.DataFrame([category_counts])

    # 收集所有batch的类别计数
    batch_counts_df = ray_dataset.map_batches(
        collect_category_counts,
        batch_format="pandas",
        batch_size=4096
    ).take_all()

    # 处理返回的DataFrame列表，提取字典
    batch_counts = []
    for df in batch_counts_df:
        if isinstance(df, pd.DataFrame):
            batch_counts.extend(df.to_dict(orient='records'))
        elif isinstance(df, dict):
            batch_counts.append(df)

    # 取最大类别数（确保覆盖所有可能类别）
    global_category_counts = {}
    for col in input_cols:
        max_count = max((batch.get(col, 0) for batch in batch_counts), default=0)
        global_category_counts[col] = max_count

    return {
        'category_counts': global_category_counts
    }


def _transform_onehotencoder(ray_dataset, input_cols: list, output_cols: list,
                           stats: Dict[str, Any], drop_last: bool, handle_invalid: str):
    """
    Transform阶段：使用类别数量进行独热编码

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        output_cols: 输出列名列表
        stats: fit阶段得到的统计量（包含category_counts）
        drop_last: 是否丢弃最后一维
        handle_invalid: 无效值处理策略

    Returns:
        Ray Dataset: 处理后的数据集
    """
    category_counts = stats.get('category_counts', {})

    def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """批处理转换函数 - 进行独热编码"""
        # 检查输入列是否存在
        existing_cols = batch.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = batch.copy()

        # 为每个输入列应用独热编码
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col not in category_counts:
                raise ValueError(f"列 {input_col} 没有类别计数信息")

            num_categories = category_counts[input_col]

            # 获取列数据
            col_data = result_df[input_col].values.astype(np.int64)

            # 处理无效值
            if handle_invalid == "error":
                invalid_mask = (col_data < 0) | (col_data >= num_categories)
                if np.any(invalid_mask):
                    invalid_values = col_data[invalid_mask]
                    raise ValueError(f"发现无效索引值: {invalid_values}")
            elif handle_invalid == "keep":
                # 将无效值设为-1，后面会创建全零向量
                col_data = np.where(
                    (col_data >= 0) & (col_data < num_categories),
                    col_data,
                    -1
                )

            # 创建独热编码矩阵
            if drop_last:
                output_dim = num_categories - 1
            else:
                output_dim = num_categories

            # 初始化独热编码矩阵
            batch_size = len(col_data)
            onehot_matrix = np.zeros((batch_size, output_dim), dtype=np.float64)

            # 为有效索引设置1
            valid_mask = (col_data >= 0) & (col_data < num_categories)
            valid_indices = col_data[valid_mask]

            if drop_last:
                # drop_last模式：第一个类别被丢弃
                # 索引0 -> 全零向量，索引1 -> [1,0,0,...]，索引2 -> [0,1,0,...]
                adjusted_indices = valid_indices - 1
                valid_adjusted_mask = adjusted_indices >= 0
                final_valid_mask = valid_mask & valid_adjusted_mask

                if np.any(final_valid_mask):
                    onehot_matrix[final_valid_mask, adjusted_indices[valid_adjusted_mask]] = 1.0
            else:
                # 正常模式：索引0 -> [1,0,0,...]，索引1 -> [0,1,0,...]
                onehot_matrix[valid_mask, valid_indices] = 1.0

            # 创建列名并添加到DataFrame
            feature_names = []
            for i in range(output_dim):
                if drop_last:
                    category_idx = i + 1  # 因为第一个类别被丢弃
                else:
                    category_idx = i
                feature_names.append(f"{output_col}_{category_idx}")

            encoded_df = pd.DataFrame(
                onehot_matrix,
                columns=feature_names,
                index=result_df.index,
                dtype=np.float64
            )

            # 将独热编码列添加到结果DataFrame
            result_df = pd.concat([result_df, encoded_df], axis=1)

        return result_df

    # 应用转换
    transformed_dataset = ray_dataset.map_batches(
        transform_batch,
        batch_format="pandas",  # 固定batch_format（遵循工程约束）
        batch_size=4096
    )

    return transformed_dataset


def run_onehotencoder_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray Data实现OneHotEncoder - 遵循fit/transform模式

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
        drop_last = spec.params.get("drop_last", True)
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"Ray OneHotEncoder: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"drop_last={drop_last}, handle_invalid={handle_invalid}")

        # 检查是否可以使用 Ray Data preprocessor
        # Ray Data 目前没有内置的OneHotEncoder preprocessor
        use_ray_preprocessor = False

        if use_ray_preprocessor:
            # 预留：如果未来Ray Data添加了OneHotEncoder支持
            pass
        else:
            # 使用自研 fit/transform 实现
            if _logger:
                _logger.debug("使用自研 fit/transform 实现")

            # Step 1: Fit阶段 - 全局统计类别数量
            stats = _fit_onehotencoder(ray_dataset, input_cols)

            if _logger:
                _logger.debug(f"Fit完成: 类别计数={stats.get('category_counts')}")

            # Step 2: Transform阶段 - 使用类别数量进行独热编码
            transformed_dataset = _transform_onehotencoder(
                ray_dataset, input_cols, output_cols, stats, drop_last, handle_invalid
            )

        return transformed_dataset

    except Exception as e:
        raise RuntimeError(f"Ray Data OneHotEncoder执行失败: {e}")


def run_onehotencoder_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    简化的OneHotEncoder实现（直接使用scikit-learn）

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame
    """
    return run_onehotencoder(input_df, spec)