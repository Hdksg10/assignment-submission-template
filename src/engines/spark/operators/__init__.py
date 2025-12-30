# Spark operators

from .standardscaler import run_standardscaler
from .imputer import run_imputer
from .tokenizer import run_tokenizer
from .hashingtf import run_hashingtf
from .idf import run_idf

# 注册算子到高性能执行器工厂
try:
    from bench.operator_executor import HighPerformanceOperatorExecutor
    HighPerformanceOperatorExecutor.register_operator('spark', 'StandardScaler', run_standardscaler)
    HighPerformanceOperatorExecutor.register_operator('spark', 'Imputer', run_imputer)
    HighPerformanceOperatorExecutor.register_operator('spark', 'Tokenizer', run_tokenizer)
    HighPerformanceOperatorExecutor.register_operator('spark', 'HashingTF', run_hashingtf)
    HighPerformanceOperatorExecutor.register_operator('spark', 'IDF', run_idf)
except ImportError:
    # 如果bench模块不可用，跳过注册（用于独立测试）
    try:
        from bench.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("bench模块不可用，跳过算子注册")
    except ImportError:
        print("WARNING: bench模块不可用，跳过算子注册")

__all__ = ['run_standardscaler', 'run_imputer', 'run_tokenizer', 'run_hashingtf', 'run_idf']