"""
Pytest配置文件

提供测试环境的配置，包括日志等级设置。
"""

import os
import logging
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """
    自动配置测试日志系统
    
    支持通过环境变量设置日志等级：
    - LOG_LEVEL: 主日志等级（默认: DEBUG）
      - DEBUG: 详细调试信息
      - INFO: 一般信息
      - WARNING: 警告信息
      - ERROR: 错误信息
      - CRITICAL: 严重错误
    
    - PY4J_LOG_LEVEL: Py4J通信日志等级（默认: WARNING，减少Spark通信日志）
      可以设置为DEBUG以查看PySpark与JVM之间的详细通信
    
    使用示例:
        LOG_LEVEL=DEBUG pytest tests/
        LOG_LEVEL=INFO PY4J_LOG_LEVEL=WARNING pytest tests/test_smoke.py
        LOG_LEVEL=DEBUG PY4J_LOG_LEVEL=DEBUG pytest tests/  # 查看所有日志包括Py4J通信
    """
    # 从环境变量获取日志等级，默认为DEBUG
    log_level_str = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
    
    # 转换为logging常量
    log_level = getattr(logging, log_level_str, logging.DEBUG)
    
    # 从环境变量获取Py4J日志等级，默认为WARNING（减少Spark通信日志）
    py4j_log_level_str = os.environ.get('PY4J_LOG_LEVEL', 'WARNING').upper()
    py4j_log_level = getattr(logging, py4j_log_level_str, logging.WARNING)
    
    # 导入并配置日志
    try:
        from bench.logger import setup_logging
        setup_logging(level=log_level, force=True, py4j_level=py4j_log_level)
    except ImportError:
        # 如果bench模块不可用，使用标准logging配置
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # 单独配置Py4J日志级别
        for logger_name in ['py4j', 'py4j.clientserver', 'py4j.java_gateway']:
            py4j_logger = logging.getLogger(logger_name)
            py4j_logger.setLevel(py4j_log_level)

