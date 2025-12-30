"""
Spark会话管理模块
"""

from typing import Optional
from pyspark.sql import SparkSession
import os
import sys

try:
    from ...bench.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


def get_spark(app_name: str = "MLBenchmark",
              master: str = "local[*]",
              config: Optional[dict] = None) -> SparkSession:
    """
    获取或创建Spark会话

    Args:
        app_name: 应用名称
        master: Master URL (默认本地模式)
        config: 额外的配置参数

    Returns:
        SparkSession: Spark会话实例
    """
    # 如果已存在活跃会话，直接返回
    if SparkSession.getActiveSession() is not None:
        return SparkSession.getActiveSession()

    # 统一指定 Python 解释器，避免 driver / worker 版本不一致
    python_exec = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

    # 构建基础配置
    spark_config = {
        "spark.app.name": app_name,
        "spark.master": master,
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.driver.memory": "2g",
        "spark.executor.memory": "2g",
        # 确保 driver / executor Python 版本一致
        "spark.pyspark.python": python_exec,
        "spark.pyspark.driver.python": python_exec,
        # 禁用不必要的日志
        "spark.sql.adaptive.logLevel": "ERROR",
    }

    # 添加用户自定义配置
    if config:
        spark_config.update(config)

    # 检查环境变量
    if "SPARK_HOME" in os.environ:
        if _logger:
            _logger.debug(f"使用SPARK_HOME: {os.environ['SPARK_HOME']}")
        else:
            print(f"DEBUG: 使用SPARK_HOME: {os.environ['SPARK_HOME']}")

    if "JAVA_HOME" in os.environ:
        if _logger:
            _logger.debug(f"使用JAVA_HOME: {os.environ['JAVA_HOME']}")
        else:
            print(f"DEBUG: 使用JAVA_HOME: {os.environ['JAVA_HOME']}")

    try:
        # 创建Spark会话构建器
        builder = SparkSession.builder

        # 应用配置
        for key, value in spark_config.items():
            builder = builder.config(key, value)

        # 创建会话
        spark = builder.getOrCreate()

        if _logger:
            _logger.info("Spark会话创建成功")
            _logger.info(f"版本: {spark.version}")
            _logger.info(f"Master: {spark.sparkContext.master}")
            _logger.info(f"应用ID: {spark.sparkContext.applicationId}")
        else:
            print("Spark会话创建成功")
            print(f"版本: {spark.version}")
            print(f"Master: {spark.sparkContext.master}")
            print(f"应用ID: {spark.sparkContext.applicationId}")

        return spark

    except Exception as e:
        raise RuntimeError(f"创建Spark会话失败: {e}")


def stop_spark(spark: SparkSession) -> None:
    """
    停止Spark会话

    Args:
        spark: Spark会话实例
    """
    if spark:
        try:
            spark.stop()
            if _logger:
                _logger.info("Spark会话已停止")
            else:
                print("Spark会话已停止")
        except Exception as e:
            if _logger:
                _logger.warning(f"停止Spark会话时出错: {e}")
            else:
                print(f"停止Spark会话时出错: {e}")


def get_spark_context_info(spark: SparkSession) -> dict:
    """
    获取Spark上下文信息

    Args:
        spark: Spark会话实例

    Returns:
        dict: 上下文信息
    """
    sc = spark.sparkContext

    return {
        "version": spark.version,
        "master": sc.master,
        "app_name": sc.appName,
        "app_id": sc.applicationId,
        "executor_count": len(sc.getConf().getAll()),
        "default_parallelism": sc.defaultParallelism,
        "python_version": sc.pythonVer,
    }


def create_spark_dataframe_from_pandas(spark: SparkSession,
                                       pandas_df,
                                       schema=None):
    """
    从pandas DataFrame创建Spark DataFrame

    Args:
        spark: Spark会话
        pandas_df: pandas DataFrame
        schema: 可选的Schema

    Returns:
        Spark DataFrame
    """
    try:
        if schema:
            df = spark.createDataFrame(pandas_df, schema=schema)
        else:
            df = spark.createDataFrame(pandas_df)

        if _logger:
            _logger.info(f"创建Spark DataFrame成功，形状: ({df.count()}, {len(df.columns)})")
        else:
            print(f"创建Spark DataFrame成功，形状: ({df.count()}, {len(df.columns)})")
        return df

    except Exception as e:
        raise RuntimeError(f"创建Spark DataFrame失败: {e}")


def convert_spark_to_pandas(spark_df, limit: Optional[int] = None):
    """
    将Spark DataFrame转换为pandas DataFrame

    Args:
        spark_df: Spark DataFrame
        limit: 限制行数（用于大文件调试）

    Returns:
        pandas DataFrame
    """
    try:
        if limit:
            pandas_df = spark_df.limit(limit).toPandas()
            if _logger:
                _logger.debug(f"转换Spark到pandas成功（限制{limit}行），形状: {pandas_df.shape}")
            else:
                print(f"DEBUG: 转换Spark到pandas成功（限制{limit}行），形状: {pandas_df.shape}")
        else:
            pandas_df = spark_df.toPandas()
            if _logger:
                _logger.debug(f"转换Spark到pandas成功，形状: {pandas_df.shape}")
            else:
                print(f"DEBUG: 转换Spark到pandas成功，形状: {pandas_df.shape}")

        return pandas_df

    except Exception as e:
        raise RuntimeError(f"转换Spark到pandas失败: {e}")
