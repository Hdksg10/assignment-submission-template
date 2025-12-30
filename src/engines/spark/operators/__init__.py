# Spark operators

from .standardscaler import run_standardscaler
from .minmaxscaler import run_minmaxscaler
from .stringindexer import run_stringindexer
from .onehotencoder import run_onehotencoder

# 注册算子到高性能执行器工厂
try:
    from bench.operator_executor import HighPerformanceOperatorExecutor
    HighPerformanceOperatorExecutor.register_operator('spark', 'StandardScaler', run_standardscaler)
    HighPerformanceOperatorExecutor.register_operator('spark', 'MinMaxScaler', run_minmaxscaler)
    HighPerformanceOperatorExecutor.register_operator('spark', 'StringIndexer', run_stringindexer)
    HighPerformanceOperatorExecutor.register_operator('spark', 'OneHotEncoder', run_onehotencoder)
except ImportError:
    # 如果bench模块不可用，跳过注册（用于独立测试）
    try:
        from bench.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("bench模块不可用，跳过算子注册")
    except ImportError:
        print("WARNING: bench模块不可用，跳过算子注册")

__all__ = ['run_standardscaler', 'run_minmaxscaler', 'run_stringindexer', 'run_onehotencoder']