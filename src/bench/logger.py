"""
日志系统配置

使用Python标准库logging模块，提供统一的日志配置和获取接口。
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


# 全局标志，确保只初始化一次
_LOGGING_INITIALIZED = False


def setup_logging(level=logging.INFO, force=False, py4j_level=None):
    """
    配置日志系统

    Args:
        level: 日志级别，默认INFO。可以是字符串（如'DEBUG', 'INFO'）或logging常量
        force: 是否强制重新配置，即使已经初始化过
        py4j_level: Py4J日志级别，默认WARNING（减少Spark通信日志）。如果为None，则使用WARNING
    """
    global _LOGGING_INITIALIZED

    # 如果已经初始化且不强制重新配置，直接返回
    if _LOGGING_INITIALIZED and not force:
        return

    # 支持字符串形式的日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # 创建logs目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 配置文件处理器（按日期轮转）
    log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 设置格式
    formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 配置根日志器
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True  # 强制重新配置，避免重复调用时的问题
    )

    # 单独配置Py4J日志级别（减少Spark通信日志）
    if py4j_level is None:
        py4j_level = logging.WARNING
    elif isinstance(py4j_level, str):
        py4j_level = getattr(logging, py4j_level.upper(), logging.WARNING)
    
    # 设置Py4J相关日志器的级别
    # Py4J日志器会传播到根日志器，但使用自己的级别进行过滤
    py4j_loggers = ['py4j', 'py4j.clientserver', 'py4j.java_gateway']
    for logger_name in py4j_loggers:
        py4j_logger = logging.getLogger(logger_name)
        py4j_logger.setLevel(py4j_level)
        # 保持传播到根日志器，这样可以使用根日志器的处理器和格式
        py4j_logger.propagate = True

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称，通常使用模块名

    Returns:
        Logger实例
    """
    return logging.getLogger(name)


# 模块导入时自动初始化日志系统
setup_logging()
