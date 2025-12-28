"""
Ray运行时管理模块
"""

import ray
import os
from typing import Optional, Dict, Any

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def init_ray(address: str = "auto",
             num_cpus: Optional[int] = None,
             num_gpus: Optional[int] = None,
             memory: Optional[int] = None,
             object_store_memory: Optional[int] = None,
             **kwargs) -> None:
    """
    初始化Ray运行时

    Args:
        address: Ray集群地址 ('auto'表示自动检测)
        num_cpus: CPU核心数
        num_gpus: GPU数量
        memory: 内存大小（字节）
        object_store_memory: 对象存储内存大小（字节）
        **kwargs: 其他Ray初始化参数
    """
    # 检查是否已经初始化
    if ray.is_initialized():
        if _logger:
            _logger.info("Ray已经初始化，跳过")
        else:
            print("Ray已经初始化，跳过")
        return

    # 从环境变量获取配置
    if num_cpus is None:
        num_cpus = int(os.environ.get("RAY_NUM_CPUS", "4"))

    if num_gpus is None:
        num_gpus = int(os.environ.get("RAY_NUM_GPUS", "0"))

    # 构建初始化参数
    init_kwargs = {
        "address": address,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "ignore_reinit_error": True,  # 允许重复初始化
        "logging_level": "WARNING",  # 减少日志输出
    }

    # 添加可选参数
    if memory:
        init_kwargs["memory"] = memory

    if object_store_memory:
        init_kwargs["object_store_memory"] = object_store_memory

    # 添加用户自定义参数
    init_kwargs.update(kwargs)

    try:
        ray.init(**init_kwargs)
        if _logger:
            _logger.info("Ray运行时初始化成功")
            _logger.info(f"版本: {ray.__version__}")
            _logger.info(f"节点数: {len(ray.nodes())}")
            _logger.info(f"CPU资源: {ray.cluster_resources().get('CPU', 0)}")
            _logger.info(f"GPU资源: {ray.cluster_resources().get('GPU', 0)}")
        else:
            print("Ray运行时初始化成功")
            print(f"版本: {ray.__version__}")
            print(f"节点数: {len(ray.nodes())}")
            print(f"CPU资源: {ray.cluster_resources().get('CPU', 0)}")
            print(f"GPU资源: {ray.cluster_resources().get('GPU', 0)}")

    except Exception as e:
        raise RuntimeError(f"Ray初始化失败: {e}")


def shutdown_ray() -> None:
    """关闭Ray运行时"""
    if ray.is_initialized():
        try:
            ray.shutdown()
            if _logger:
                _logger.info("Ray运行时已关闭")
            else:
                print("Ray运行时已关闭")
        except Exception as e:
            if _logger:
                _logger.warning(f"关闭Ray运行时时出错: {e}")
            else:
                print(f"关闭Ray运行时时出错: {e}")
    else:
        if _logger:
            _logger.debug("Ray未初始化，无需关闭")
        else:
            print("Ray未初始化，无需关闭")


def get_ray_context_info() -> Dict[str, Any]:
    """
    获取Ray上下文信息

    Returns:
        dict: Ray运行时信息
    """
    if not ray.is_initialized():
        return {"status": "not_initialized"}

    try:
        nodes = ray.nodes()
        resources = ray.cluster_resources()

        return {
            "status": "initialized",
            "version": ray.__version__,
            "nodes": len(nodes),
            "resources": {
                "cpu": resources.get("CPU", 0),
                "gpu": resources.get("GPU", 0),
                "memory": resources.get("memory", 0),
                "object_store_memory": resources.get("object_store_memory", 0),
            },
            "alive_nodes": sum(1 for node in nodes if node["alive"]),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def create_ray_dataset_from_pandas(pandas_df):
    """
    从pandas DataFrame创建Ray Dataset

    Args:
        pandas_df: pandas DataFrame

    Returns:
        Ray Dataset
    """
    try:
        import ray.data as rd

        dataset = rd.from_pandas(pandas_df)
        if _logger:
            _logger.info(f"创建Ray Dataset成功，行数: {dataset.count()}")
        else:
            print(f"创建Ray Dataset成功，行数: {dataset.count()}")

        return dataset

    except Exception as e:
        raise RuntimeError(f"创建Ray Dataset失败: {e}")


def convert_ray_dataset_to_pandas(ray_dataset):
    """
    将Ray Dataset转换为pandas DataFrame

    Args:
        ray_dataset: Ray Dataset

    Returns:
        pandas DataFrame
    """
    try:
        pandas_df = ray_dataset.to_pandas()
        if _logger:
            _logger.debug(f"转换Ray Dataset到pandas成功，形状: {pandas_df.shape}")
        else:
            print(f"DEBUG: 转换Ray Dataset到pandas成功，形状: {pandas_df.shape}")

        return pandas_df

    except Exception as e:
        raise RuntimeError(f"转换Ray Dataset到pandas失败: {e}")


def create_ray_dataset_from_csv(path: str, **kwargs):
    """
    直接从CSV文件创建Ray Dataset

    Args:
        path: CSV文件路径
        **kwargs: 读取参数

    Returns:
        Ray Dataset
    """
    try:
        import ray.data as rd

        # 设置默认参数
        read_kwargs = {
            "filesystem": None,
            "parallelism": -1,  # 自动并行度
        }
        read_kwargs.update(kwargs)

        dataset = rd.read_csv(path, **read_kwargs)
        if _logger:
            _logger.info(f"从CSV创建Ray Dataset成功，行数: {dataset.count()}")
        else:
            print(f"从CSV创建Ray Dataset成功，行数: {dataset.count()}")

        return dataset

    except Exception as e:
        raise RuntimeError(f"从CSV创建Ray Dataset失败: {e}")
