# Ray operators

from .standardscaler import run_standardscaler_with_ray_data
from .minmaxscaler import run_minmaxscaler_with_ray_data
from .stringindexer import run_stringindexer_with_ray_data
from .onehotencoder import run_onehotencoder_with_ray_data

# 为了兼容性保留旧接口
from .standardscaler import run_standardscaler
from .minmaxscaler import run_minmaxscaler
from .stringindexer import run_stringindexer
from .onehotencoder import run_onehotencoder

# 注册算子到高性能执行器工厂
try:
    from bench.operator_executor import HighPerformanceOperatorExecutor
    HighPerformanceOperatorExecutor.register_operator('ray', 'StandardScaler', run_standardscaler_with_ray_data)
    HighPerformanceOperatorExecutor.register_operator('ray', 'MinMaxScaler', run_minmaxscaler_with_ray_data)
    HighPerformanceOperatorExecutor.register_operator('ray', 'StringIndexer', run_stringindexer_with_ray_data)
    HighPerformanceOperatorExecutor.register_operator('ray', 'OneHotEncoder', run_onehotencoder_with_ray_data)
except ImportError:
    # 如果bench模块不可用，跳过注册（用于独立测试）
    try:
        from ...bench.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("bench模块不可用，跳过算子注册")
    except ImportError:
        print("WARNING: bench模块不可用，跳过算子注册")

__all__ = [
    'run_standardscaler', 'run_standardscaler_with_ray_data',
    'run_minmaxscaler', 'run_minmaxscaler_with_ray_data',
    'run_stringindexer', 'run_stringindexer_with_ray_data',
    'run_onehotencoder', 'run_onehotencoder_with_ray_data'
]
