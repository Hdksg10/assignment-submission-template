"""
统一的分布式执行触发工具

提供轻量级的action操作来触发Spark/Ray的lazy执行，
不引入数据收集到driver的开销。
"""

from typing import Any


def materialize_spark(df: Any) -> None:
    """
    触发Spark DataFrame的执行
    
    使用count()在JVM内执行，不引入Python foreachPartition的额外开销。
    
    Args:
        df: Spark DataFrame
    """
    if hasattr(df, 'count'):
        df.count()
    else:
        raise TypeError(f"Expected Spark DataFrame, got {type(df)}")


def materialize_ray(ds: Any) -> None:
    """
    触发Ray Dataset的执行
    
    先materialize()再count()，避免重复计算。
    
    Args:
        ds: Ray Dataset
    """
    if hasattr(ds, 'materialize'):
        # materialize() 确保数据被物化到内存/磁盘
        materialized = ds.materialize()
        # count() 触发执行（如果还没执行的话）
        materialized.count()
    elif hasattr(ds, 'count'):
        # 如果没有materialize方法，至少调用count
        ds.count()
    else:
        raise TypeError(f"Expected Ray Dataset, got {type(ds)}")

