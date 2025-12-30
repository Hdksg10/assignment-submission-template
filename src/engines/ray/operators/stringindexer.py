"""
Ray StringIndexer算子实现

遵循设计原则：Stateful算子必须拆成 fit → transform 模式。

实现路径优先级（按设计原则）：
1. 优先：Ray Data preprocessor（若可用）
2. 其次：自研 fit/transform
   - 先全局统计类别集合并建立映射
   - 再使用映射进行transform
   - 支持handle_invalid策略：error/skip/keep
"""

import ray
import ray.data
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


def run_stringindexer(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    兼容性函数：直接处理pandas DataFrame

    主要用于测试和兼容性，生产环境建议使用run_stringindexer_with_ray_data
    此函数使用scikit-learn的LabelEncoder，确保与fit/transform模式语义一致。

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame，包含索引后的列（dtype=int64）
    """
    try:
        from sklearn.preprocessing import LabelEncoder

        # 获取参数
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"Ray StringIndexer (pandas兼容模式): input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"handle_invalid={handle_invalid}")
        else:
            print(f"DEBUG: Ray StringIndexer参数: input_cols={input_cols}, output_cols={output_cols}")
            print(f"DEBUG: handle_invalid={handle_invalid}")

        # 检查输入列是否存在
        existing_cols = input_df.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = input_df.copy()

        # 为每个输入列创建LabelEncoder并应用
        for input_col, output_col in zip(input_cols, output_cols):
            # 获取列数据
            col_data = result_df[input_col]

            # 处理缺失值（转换为字符串）
            col_data = col_data.fillna('<MISSING>').astype(str)

            # 创建LabelEncoder并fit
            encoder = LabelEncoder()
            encoder.fit(col_data)

            # Transform
            try:
                encoded_data = encoder.transform(col_data)
                result_df[output_col] = encoded_data.astype(np.int64)
            except ValueError as e:
                if handle_invalid == "error":
                    raise ValueError(f"发现未知类别值: {e}")
                elif handle_invalid == "skip":
                    # 过滤掉包含无效值的行
                    valid_mask = col_data.isin(encoder.classes_)
                    result_df = result_df[valid_mask].copy()
                    encoded_data = encoder.transform(col_data[valid_mask])
                    result_df[output_col] = encoded_data.astype(np.int64)
                elif handle_invalid == "keep":
                    # 将未知值映射到特殊索引（classes数量）
                    encoded_data = np.full(len(col_data), -1, dtype=np.int64)
                    known_mask = col_data.isin(encoder.classes_)
                    encoded_data[known_mask] = encoder.transform(col_data[known_mask])
                    # 未知值使用最大索引+1
                    unknown_index = len(encoder.classes_)
                    encoded_data[~known_mask] = unknown_index
                    result_df[output_col] = encoded_data.astype(np.int64)
                else:
                    raise ValueError(f"不支持的handle_invalid策略: {handle_invalid}")

        if _logger:
            _logger.info(f"Ray StringIndexer处理完成，输出列: {result_df.columns.tolist()}")
        else:
            print(f"Ray StringIndexer处理完成，输出列: {result_df.columns.tolist()}")

        # 验证输出
        if len(output_cols) != len(input_cols):
            raise ValueError("输出列数量与输入列数量不匹配")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Ray StringIndexer执行失败: {e}")


def _fit_stringindexer(ray_dataset, input_cols: list) -> Dict[str, Any]:
    """
    Fit阶段：全局统计类别集合并建立映射

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表

    Returns:
        Dict包含类别映射信息
    """

    def collect_categories(batch: pd.DataFrame) -> Dict[str, Any]:
        """收集单个batch的类别"""
        categories = {}
        for col in input_cols:
            if col in batch.columns:
                # 处理缺失值并转换为字符串
                col_data = batch[col].fillna('<MISSING>').astype(str)
                unique_vals = col_data.unique().tolist()
                categories[col] = unique_vals
        return categories

    # 收集所有batch的类别
    batch_categories = ray_dataset.map_batches(
        collect_categories,
        batch_format="pandas",
        batch_size=4096,
        compute=ray.data.TaskPoolStrategy()
    ).take_all()

    # 合并所有batch的类别集合
    global_categories = {}
    for col in input_cols:
        all_cats = set()
        for batch_cats in batch_categories:
            if col in batch_cats:
                all_cats.update(batch_cats[col])
        global_categories[col] = sorted(list(all_cats))  # 排序确保一致性

    # 建立索引映射
    category_mappings = {}
    for col, categories in global_categories.items():
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        category_mappings[col] = mapping

    return {
        'category_mappings': category_mappings,
        'global_categories': global_categories
    }


def _transform_stringindexer(ray_dataset, input_cols: list, output_cols: list,
                           stats: Dict[str, Any], handle_invalid: str):
    """
    Transform阶段：使用全局类别映射进行索引转换

    Args:
        ray_dataset: Ray Dataset
        input_cols: 输入列名列表
        output_cols: 输出列名列表
        stats: fit阶段得到的统计量（包含category_mappings）
        handle_invalid: 无效值处理策略

    Returns:
        Ray Dataset: 处理后的数据集
    """
    category_mappings = stats.get('category_mappings', {})

    def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """批处理转换函数 - 使用全局类别映射"""
        # 检查输入列是否存在
        existing_cols = batch.columns.tolist()
        missing_cols = [col for col in input_cols if col not in existing_cols]
        if missing_cols:
            raise ValueError(f"输入列不存在: {missing_cols}")

        # 复制DataFrame避免修改原数据（遵循工程约束）
        result_df = batch.copy()

        # 为每个输入列应用映射
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col not in category_mappings:
                raise ValueError(f"列 {input_col} 没有类别映射")

            mapping = category_mappings[input_col]

            # 处理缺失值并转换为字符串
            col_data = result_df[input_col].fillna('<MISSING>').astype(str)

            # 应用映射
            if handle_invalid == "error":
                # 检查是否有未知类别
                unknown_cats = set(col_data) - set(mapping.keys())
                if unknown_cats:
                    raise ValueError(f"发现未知类别值: {unknown_cats}")
                encoded_data = col_data.map(mapping).astype(np.int64)

            elif handle_invalid == "skip":
                # 过滤掉包含无效值的行
                valid_mask = col_data.isin(mapping.keys())
                result_df = result_df[valid_mask].copy()
                encoded_data = col_data[valid_mask].map(mapping).astype(np.int64)

            elif handle_invalid == "keep":
                # 将未知值映射到特殊索引
                encoded_data = np.full(len(col_data), -1, dtype=np.int64)
                known_mask = col_data.isin(mapping.keys())
                encoded_data[known_mask] = col_data[known_mask].map(mapping).astype(np.int64)
                # 未知值使用最大索引+1
                unknown_index = len(mapping)
                encoded_data[~known_mask] = unknown_index
                encoded_data = encoded_data.astype(np.int64)

            else:
                raise ValueError(f"不支持的handle_invalid策略: {handle_invalid}")

            result_df[output_col] = encoded_data

        return result_df

    # 应用转换
    transformed_dataset = ray_dataset.map_batches(
        transform_batch,
        batch_format="pandas",  # 固定batch_format（遵循工程约束）
        batch_size=4096
    )

    return transformed_dataset


def run_stringindexer_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray Data实现StringIndexer - 遵循fit/transform模式

    核心设计原则：
    1. Stateful算子必须先全局fit，再transform
    2. 禁止在map_batches内fit（避免每个batch各自fit的错误语义）
    3. 输出dtype统一为int64
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
        handle_invalid = spec.params.get("handle_invalid", "error")

        if _logger:
            _logger.debug(f"Ray StringIndexer: input_cols={input_cols}, output_cols={output_cols}")
            _logger.debug(f"handle_invalid={handle_invalid}")

        # 检查是否可以使用 Ray Data preprocessor
        # Ray Data 目前没有内置的StringIndexer preprocessor
        use_ray_preprocessor = False

        if use_ray_preprocessor:
            # 预留：如果未来Ray Data添加了StringIndexer支持
            pass
        else:
            # 使用自研 fit/transform 实现
            if _logger:
                _logger.debug("使用自研 fit/transform 实现")

            # Step 1: Fit阶段 - 全局统计类别集合
            stats = _fit_stringindexer(ray_dataset, input_cols)

            if _logger:
                _logger.debug(f"Fit完成: 类别映射已建立")

            # Step 2: Transform阶段 - 使用全局类别映射进行索引
            transformed_dataset = _transform_stringindexer(
                ray_dataset, input_cols, output_cols, stats, handle_invalid
            )

        return transformed_dataset

    except Exception as e:
        raise RuntimeError(f"Ray Data StringIndexer执行失败: {e}")


def run_stringindexer_simple(input_df: pd.DataFrame, spec: OperatorSpec) -> pd.DataFrame:
    """
    简化的StringIndexer实现（直接使用scikit-learn）

    Args:
        input_df: 输入pandas DataFrame
        spec: 算子规格

    Returns:
        DataFrame: 输出DataFrame
    """
    return run_stringindexer(input_df, spec)