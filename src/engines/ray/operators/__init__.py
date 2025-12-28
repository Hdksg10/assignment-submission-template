# Ray operators

from .standardscaler import run_standardscaler_with_ray_data

# 为了兼容性保留旧接口
from .standardscaler import run_standardscaler

# 注册算子到高性能执行器工厂
try:
    from bench.operator_executor import HighPerformanceOperatorExecutor
    HighPerformanceOperatorExecutor.register_operator('ray', 'StandardScaler', run_standardscaler_with_ray_data)
except ImportError:
    # 如果bench模块不可用，跳过注册（用于独立测试）
    try:
        from ...bench.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("bench模块不可用，跳过算子注册")
    except ImportError:
        print("WARNING: bench模块不可用，跳过算子注册")

__all__ = ['run_standardscaler', 'run_standardscaler_with_ray_data']
