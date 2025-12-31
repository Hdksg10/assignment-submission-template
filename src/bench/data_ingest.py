"""
数据读取模块 - 支持多机集群的原生引擎读取

提供统一的数据加载接口，支持：
- pandas 模式：单机小数据，driver 先读全量
- engine 模式：多机大数据，使用引擎原生读取（Spark read.csv、Ray read_csv）
"""

from typing import Union, Optional
from pathlib import Path
import pandas as pd
import os

from .logger import get_logger

logger = get_logger(__name__)


def _is_shared_storage_path(path: str) -> bool:
    """
    检查路径是否为共享存储路径（适用于多机集群）
    
    Args:
        path: 文件路径
    
    Returns:
        True 如果是共享存储路径，False 否则
    """
    path_lower = path.lower()
    
    # 检查常见的共享存储协议和路径
    shared_storage_indicators = [
        "hdfs://",           # HDFS
        "s3://",             # AWS S3
        "s3a://",            # AWS S3 (Hadoop兼容)
        "s3n://",            # AWS S3 (旧版)
        "gs://",             # Google Cloud Storage
        "abfs://",           # Azure Blob Storage
        "abfss://",          # Azure Blob Storage (安全)
        "wasb://",           # Azure Blob Storage (旧版)
        "adl://",            # Azure Data Lake
        "file://",           # 文件协议（可能是NFS挂载）
    ]
    
    # 检查是否以共享存储协议开头
    for indicator in shared_storage_indicators:
        if path_lower.startswith(indicator):
            return True
    
    # 检查是否为绝对路径且可能是NFS挂载点
    # 常见的NFS挂载点（可根据实际情况调整）
    common_nfs_mounts = ["/mnt/", "/shared/", "/data/", "/nfs/", "/hdfs/"]
    abs_path = os.path.abspath(path)
    for mount in common_nfs_mounts:
        if abs_path.startswith(mount):
            # 进一步检查：如果是绝对路径且不在用户主目录下，可能是共享存储
            home_dir = os.path.expanduser("~")
            if not abs_path.startswith(home_dir):
                return True
    
    return False


def _validate_data_path_for_distributed(path: str, is_distributed: bool) -> None:
    """
    验证数据路径是否适合分布式环境
    
    Args:
        path: 数据文件路径
        is_distributed: 是否为分布式模式（多机集群）
    
    Raises:
        ValueError: 如果路径不适合分布式环境
    """
    if not is_distributed:
        return
    
    if not _is_shared_storage_path(path):
        # 检查是否为本地路径
        if os.path.isabs(path) and not path.startswith(("http://", "https://")):
            # 检查文件是否存在（仅在driver节点检查）
            if os.path.exists(path):
                warning_msg = (
                    f"⚠️  警告：检测到本地文件路径 '{path}' 在多机模式下使用。\n"
                    "在多机集群中，worker节点可能无法访问本地路径。\n"
                    "建议使用共享存储路径，例如：\n"
                    "  - HDFS: hdfs://namenode:port/path/to/file.csv\n"
                    "  - S3: s3://bucket/path/to/file.csv\n"
                    "  - NFS: /shared/data/file.csv (确保所有节点可访问)\n"
                    "  - 或其他分布式文件系统路径"
                )
                logger.warning(warning_msg)
            else:
                error_msg = (
                    f"错误：在多机模式下，路径 '{path}' 不存在或无法访问。\n"
                    "请确保：\n"
                    "  1. 使用共享存储路径（HDFS、S3、NFS等）\n"
                    "  2. 所有节点都能访问该路径\n"
                    "  3. 路径格式正确"
                )
                raise ValueError(error_msg)


def load_input_for_engine(engine: str, path: str, spark=None, is_distributed: bool = True):
    """
    为指定引擎加载输入数据（引擎原生读取，适用于多机集群）

    Args:
        engine: 引擎名称 ('spark' 或 'ray')
        path: 数据文件路径（必须是所有节点可访问的共享存储路径）
        spark: SparkSession 实例（仅当 engine='spark' 时需要）
        is_distributed: 是否为分布式模式（默认True，用于路径验证）

    Returns:
        Spark DataFrame 或 Ray Dataset

    Raises:
        ValueError: 如果引擎不支持或参数无效，或路径不适合分布式环境
    """
    # 验证路径是否适合分布式环境
    _validate_data_path_for_distributed(path, is_distributed)
    
    if engine == "spark":
        if spark is None:
            raise ValueError("Spark 引擎需要提供 spark 参数")
        
        logger.info(f"使用 Spark 原生读取: {path}")
        # 使用 Spark 原生读取，支持分布式文件系统（HDFS、S3、NFS等）
        df = spark.read.csv(
            path,
            header=True,
            inferSchema=True
        )
        logger.info(f"Spark DataFrame 创建成功，行数: {df.count()}")
        return df
        
    elif engine == "ray":
        import ray.data as rd
        
        logger.info(f"使用 Ray 原生读取: {path}")
        # 使用 Ray 原生读取，支持分布式文件系统
        dataset = rd.read_csv(path)
        logger.info(f"Ray Dataset 创建成功，行数: {dataset.count()}")
        return dataset
        
    else:
        raise ValueError(f"不支持的引擎: {engine}。支持: 'spark', 'ray'")


def load_input_pandas(path: str) -> pd.DataFrame:
    """
    使用 pandas 加载数据（单机模式，driver 先读全量）

    Args:
        path: 数据文件路径

    Returns:
        pandas DataFrame
    """
    from .io import load_csv
    
    logger.info(f"使用 pandas 读取: {path}")
    df = load_csv(path)
    logger.info(f"pandas DataFrame 加载成功，形状: {df.shape}")
    return df

