"""
数据IO模块

提供统一的数据加载和保存接口。
"""

import pandas as pd
import os
from typing import Optional, Union
from pathlib import Path

from .logger import get_logger


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    加载CSV文件

    Args:
        path: 文件路径
        **kwargs: pandas.read_csv的参数

    Returns:
        DataFrame: 加载的数据
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 设置默认参数
    defaults = {
        'sep': ',',
        'header': 0,
        'index_col': None,
        'encoding': 'utf-8'
    }
    defaults.update(kwargs)

    logger = get_logger(__name__)
    try:
        df = pd.read_csv(path, **defaults)
        logger.info(f"成功加载CSV文件: {path} (形状: {df.shape})")
        return df
    except Exception as e:
        raise RuntimeError(f"加载CSV文件失败: {path}, 错误: {e}")


def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    保存DataFrame为CSV文件

    Args:
        df: 要保存的DataFrame
        path: 保存路径
        **kwargs: pandas.to_csv的参数
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 设置默认参数
    defaults = {
        'sep': ',',
        'index': False,
        'encoding': 'utf-8'
    }
    defaults.update(kwargs)

    logger = get_logger(__name__)
    try:
        df.to_csv(path, **defaults)
        logger.info(f"成功保存CSV文件: {path} (形状: {df.shape})")
    except Exception as e:
        raise RuntimeError(f"保存CSV文件失败: {path}, 错误: {e}")


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    加载Parquet文件

    Args:
        path: 文件路径
        **kwargs: pandas.read_parquet的参数

    Returns:
        DataFrame: 加载的数据
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    logger = get_logger(__name__)
    try:
        df = pd.read_parquet(path, **kwargs)
        logger.info(f"成功加载Parquet文件: {path} (形状: {df.shape})")
        return df
    except Exception as e:
        raise RuntimeError(f"加载Parquet文件失败: {path}, 错误: {e}")


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    保存DataFrame为Parquet文件

    Args:
        df: 要保存的DataFrame
        path: 保存路径
        **kwargs: pandas.to_parquet的参数
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 设置默认参数
    defaults = {
        'index': False
    }
    defaults.update(kwargs)

    logger = get_logger(__name__)
    try:
        df.to_parquet(path, **defaults)
        logger.info(f"成功保存Parquet文件: {path} (形状: {df.shape})")
    except Exception as e:
        raise RuntimeError(f"保存Parquet文件失败: {path}, 错误: {e}")


def get_file_info(path: Union[str, Path]) -> dict:
    """
    获取文件基本信息

    Args:
        path: 文件路径

    Returns:
        dict: 包含文件大小、行数、列数等信息
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 获取文件大小
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # 尝试读取基本信息
    try:
        if path.suffix.lower() == '.csv':
            # 读取CSV头几行获取列数
            df_sample = pd.read_csv(path, nrows=5)
            columns = list(df_sample.columns)
            dtypes = df_sample.dtypes.to_dict()
        elif path.suffix.lower() == '.parquet':
            df_sample = pd.read_parquet(path)
            columns = list(df_sample.columns)
            dtypes = df_sample.dtypes.to_dict()
        else:
            columns = None
            dtypes = None

        return {
            'path': str(path),
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'columns': columns,
            'dtypes': dtypes
        }
    except Exception as e:
        return {
            'path': str(path),
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'error': str(e)
        }


def validate_dataframe(df: pd.DataFrame,
                      required_cols: Optional[list] = None,
                      expected_dtypes: Optional[dict] = None) -> list:
    """
    验证DataFrame的结构

    Args:
        df: 要验证的DataFrame
        required_cols: 必需的列名列表
        expected_dtypes: 期望的数据类型字典

    Returns:
        list: 验证错误信息列表，空列表表示验证通过
    """
    errors = []

    # 检查必需列
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"缺少必需列: {missing_cols}")

    # 检查数据类型
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    errors.append(f"列 '{col}' 类型不匹配: 期望 {expected_dtype}, 实际 {actual_dtype}")

    # 检查数据完整性
    if df.empty:
        errors.append("DataFrame为空")

    return errors
