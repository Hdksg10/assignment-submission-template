"""
Ray运行时管理模块
"""

import ray
import os
import socket
from typing import Optional, Dict, Any

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def _normalize_ray_address(address: Optional[str]) -> Optional[str]:
    """
    规范化Ray地址以便比较
    
    Args:
        address: Ray地址（None, 'auto', 或 '<host>:<port>'）
    
    Returns:
        规范化后的地址字符串
    """
    if address is None:
        return None
    if address == "auto":
        return "auto"
    # 对于具体地址，提取host和port
    if ":" in address:
        host, port = address.rsplit(":", 1)
        return f"{host}:{port}"
    return address


def _validate_ray_cluster_connection(address: Optional[str], 
                                     node_count: int = None,
                                     alive_node_count: int = None) -> None:
    """
    验证Ray多机模式连接
    
    Args:
        address: Ray集群地址
        node_count: 节点总数（如果已获取）
        alive_node_count: 活跃节点数（如果已获取）
    
    Raises:
        RuntimeError: 如果多机模式验证失败
    """
    if address is None:
        return
    
    # 如果提供了节点信息，验证节点数
    if node_count is not None:
        if node_count <= 1:
            warning_msg = (
                f"⚠️  警告：指定了多机模式 (address={address})，但只检测到 {node_count} 个节点。"
                "这可能表示：\n"
                "  1. 集群尚未完全启动\n"
                "  2. 网络连接问题\n"
                "  3. 实际运行在本地模式"
            )
            if _logger:
                _logger.warning(warning_msg)
            else:
                print(warning_msg)
        elif alive_node_count is not None and alive_node_count < node_count:
            warning_msg = (
                f"⚠️  警告：检测到 {node_count} 个节点，但只有 {alive_node_count} 个活跃节点。"
                "部分节点可能已离线。"
            )
            if _logger:
                _logger.warning(warning_msg)
            else:
                print(warning_msg)
    
    # 验证地址格式和可达性（仅对具体地址，不包括'auto'）
    if address != "auto" and ":" in address:
        try:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
            
            # 尝试连接以验证可达性
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2秒超时
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                warning_msg = (
                    f"⚠️  警告：无法连接到Ray集群地址 {address}。"
                    "请检查：\n"
                    "  1. 集群是否已启动\n"
                    "  2. 网络连接是否正常\n"
                    "  3. 防火墙设置是否正确"
                )
                if _logger:
                    _logger.warning(warning_msg)
                else:
                    print(warning_msg)
        except (ValueError, socket.gaierror, socket.timeout) as e:
            warning_msg = (
                f"⚠️  警告：验证Ray集群地址 {address} 时出错: {e}。"
                "请确认地址格式正确（应为 'host:port' 或 'auto'）"
            )
            if _logger:
                _logger.warning(warning_msg)
            else:
                print(warning_msg)


def init_ray(address: Optional[str] = None,
             namespace: str = "benchmark",
             runtime_env_json: Optional[str] = None,
             num_cpus: Optional[int] = None,
             num_gpus: Optional[int] = None,
             memory: Optional[int] = None,
             object_store_memory: Optional[int] = None,
             **kwargs) -> None:
    """
    初始化Ray运行时

    Args:
        address: Ray集群地址 (None 表示本地模式，'auto' 或 '<head-ip>:6379' 表示多机集群)
        namespace: Ray命名空间（默认: benchmark）
        runtime_env_json: 运行时环境 JSON 字符串（用于打包代码依赖等）
        num_cpus: CPU核心数
        num_gpus: GPU数量
        memory: 内存大小（字节）
        object_store_memory: 对象存储内存大小（字节）
        **kwargs: 其他Ray初始化参数
    """
    # 检查是否已经初始化，并验证集群连接
    if ray.is_initialized():
        try:
            # 获取当前连接的集群地址
            current_address = ray.get_runtime_context().gcs_address
            # 规范化地址格式以便比较
            current_address_normalized = _normalize_ray_address(current_address)
            target_address_normalized = _normalize_ray_address(address)
            
            # 如果目标地址与当前地址不同，需要重新初始化
            if target_address_normalized != current_address_normalized:
                if _logger:
                    _logger.warning(
                        f"Ray已连接到不同集群 (当前: {current_address}, 目标: {address})，"
                        "将关闭当前连接并重新初始化"
                    )
                else:
                    print(f"WARNING: Ray已连接到不同集群 (当前: {current_address}, 目标: {address})，"
                          "将关闭当前连接并重新初始化")
                ray.shutdown()
            else:
                # 验证多机模式连接（如果指定了address）
                if address is not None:
                    try:
                        nodes = ray.nodes()
                        node_count = len(nodes)
                        alive_node_count = sum(1 for node in nodes if node.get("alive", False))
                        _validate_ray_cluster_connection(address, node_count, alive_node_count)
                    except Exception as e:
                        if _logger:
                            _logger.warning(f"验证集群连接时出错: {e}")
                
                if _logger:
                    _logger.info("Ray已经初始化，且连接到正确的集群，跳过")
                else:
                    print("Ray已经初始化，且连接到正确的集群，跳过")
                return
        except Exception as e:
            # 如果获取地址失败，关闭并重新初始化
            if _logger:
                _logger.warning(f"获取Ray集群地址失败: {e}，将重新初始化")
            else:
                print(f"WARNING: 获取Ray集群地址失败: {e}，将重新初始化")
            try:
                ray.shutdown()
            except:
                pass

    # 从环境变量获取配置
    if num_cpus is None:
        num_cpus = int(os.environ.get("RAY_NUM_CPUS", "4"))

    if num_gpus is None:
        num_gpus = int(os.environ.get("RAY_NUM_GPUS", "0"))

    # 解析 runtime_env（如果提供）
    runtime_env = None
    if runtime_env_json:
        try:
            import json
            runtime_env = json.loads(runtime_env_json)
            if _logger:
                _logger.debug(f"解析 runtime_env: {runtime_env}")
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的 runtime_env_json: {e}")

    # 构建初始化参数
    init_kwargs = {
        "namespace": namespace,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "ignore_reinit_error": True,  # 允许重复初始化
        "logging_level": "WARNING",  # 减少日志输出
    }

    # 设置 address（如果提供）
    if address is not None:
        init_kwargs["address"] = address
        if _logger:
            _logger.info(f"多机模式：连接到 Ray 集群 address={address}")

    # 设置 runtime_env（如果提供）
    if runtime_env is not None:
        init_kwargs["runtime_env"] = runtime_env

    # 添加可选参数
    if memory:
        init_kwargs["memory"] = memory

    if object_store_memory:
        init_kwargs["object_store_memory"] = object_store_memory

    # 添加用户自定义参数
    init_kwargs.update(kwargs)

    try:
        ray.init(**init_kwargs)
        
        # 打印集群连接信息（用于确认多机连接）
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()
        node_count = len(nodes)
        alive_node_count = sum(1 for node in nodes if node.get("alive", False))
        
        # 验证多机模式连接（如果指定了address）
        if address is not None:
            _validate_ray_cluster_connection(address, node_count, alive_node_count)
        
        if _logger:
            _logger.info("Ray运行时初始化成功")
            _logger.info(f"版本: {ray.__version__}")
            _logger.info(f"命名空间: {namespace}")
            if address:
                _logger.info(f"集群地址: {address}")
                _logger.info("✓ 已连接到 Ray 集群（多机模式）")
            else:
                _logger.info("本地模式")
            _logger.info(f"节点数: {node_count} (活跃: {alive_node_count})")
            _logger.info(f"集群资源: {cluster_resources}")
            _logger.info(f"CPU资源: {cluster_resources.get('CPU', 0)}")
            _logger.info(f"GPU资源: {cluster_resources.get('GPU', 0)}")
        else:
            print("Ray运行时初始化成功")
            print(f"版本: {ray.__version__}")
            print(f"命名空间: {namespace}")
            if address:
                print(f"集群地址: {address}")
                print("✓ 已连接到 Ray 集群（多机模式）")
            else:
                print("本地模式")
            print(f"节点数: {node_count} (活跃: {alive_node_count})")
            print(f"集群资源: {cluster_resources}")
            print(f"CPU资源: {cluster_resources.get('CPU', 0)}")
            print(f"GPU资源: {cluster_resources.get('GPU', 0)}")

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
