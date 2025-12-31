"""
统一的分布式执行触发工具

通过输出列的 checksum 聚合来强制真实计算，避免：
- Spark: count() 触发 column pruning，transform 被优化掉
- Ray: materialize() 把物化/缓存写入开销算进算子本体
"""

from __future__ import annotations

from typing import Any, Sequence


def force_execute(obj: Any, *, engine: str, cols: Sequence[str]) -> float:
    """
    强制真实计算，依赖输出列的值
    
    Args:
        obj: Spark DataFrame 或 Ray Dataset
        engine: 引擎名称 ('spark' 或 'ray')
        cols: 要触发的输出列列表（必须依赖这些列的值）
    
    Returns:
        float: checksum 值（用于验证是否依赖输出列）
    """
    if engine == "spark":
        return _force_execute_spark(obj, cols)
    elif engine == "ray":
        return _force_execute_ray(obj, cols)
    else:
        raise ValueError(f"Unknown engine={engine}")


def _force_execute_spark(df: Any, cols: Sequence[str]) -> float:
    """
    Spark: 用聚合 checksum，强制计算输出列
    
    使用输出列的聚合表达式，确保 Spark Catalyst 不会把 transform 计算优化掉
    
    Returns:
        float: checksum 值
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType
    
    if not cols:
        # 兜底：至少别 crash，但不推荐（可能裁剪）
        count = df.count()
        return float(count)  # 返回行数作为 checksum（不依赖列值）
    
    schema = {f.name: f.dataType for f in df.schema.fields}
    exprs = []
    
    for c in cols:
        if c not in schema:
            continue
        
        dt = schema[c]
        col = F.col(c)
        
        # 数值列：sum 最轻
        if isinstance(dt, NumericType):
            exprs.append(F.sum(col).cast("double"))
        # 字符串列：sum(length) 轻且依赖值
        elif isinstance(dt, StringType):
            exprs.append(F.sum(F.length(col)).cast("double"))
        else:
            # 其他复杂类型（vector/array/struct/udt）：用 length(cast(string))
            # 依赖值，能防止 column pruning；比 to_json/展开更稳、实现更简单
            exprs.append(F.sum(F.length(col.cast("string"))).cast("double"))
    
    if not exprs:
        count = df.count()
        return float(count)
    
    # 把多个 expr 合成一个，collect 单行，避免大规模回传
    agg = exprs[0]
    for e in exprs[1:]:
        agg = agg + e
    
    result = df.agg(agg.alias("__chk")).collect()
    if result and len(result) > 0:
        chk_value = result[0]["__chk"]
        return float(chk_value) if chk_value is not None else 0.0
    return 0.0


def _force_execute_ray(ds: Any, cols: Sequence[str]) -> float:
    """
    Ray: 用 per-batch checksum + 小量回传（不要 materialize）
    
    不做 materialize()，避免把"写对象存储/缓存"的成本掺进测量；
    同时 checksum 依赖输出列，保证 map/transform 真被执行
    
    Returns:
        float: checksum 值
    """
    import pandas as pd
    import numpy as np
    
    if not cols:
        count = ds.count()
        return float(count)  # 返回行数作为 checksum（不依赖列值）
    
    def batch_checksum(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[list(cols)].copy()
        
        # 对 object 列做一个"便宜但依赖值"的签名，避免 repr/深拷贝过重
        for c in sub.columns:
            if sub[c].dtype == "object":
                # 常见：list[str]/csr_matrix/ndarray 等
                def sig(x):
                    if x is None:
                        return 0
                    # scipy sparse：nnz O(1)，足以依赖值(非零数)
                    nnz = getattr(x, "nnz", None)
                    if nnz is not None:
                        return int(nnz)
                    # ndarray/list：长度依赖值结构（足以触发 transform 的创建）
                    if hasattr(x, "__len__"):
                        try:
                            return int(len(x))
                        except Exception:
                            return 1
                    return 1
                
                sub[c] = sub[c].map(sig)
        
        h = pd.util.hash_pandas_object(sub, index=False).astype("uint64")
        chk = int(h.sum() & np.uint64(0xFFFFFFFFFFFFFFFF))
        return pd.DataFrame({"__chk": [chk]})
    
    chk_ds = ds.map_batches(batch_checksum, batch_format="pandas")
    # 回传很小（每个 block 1 行），用 take_all 合并即可
    parts = chk_ds.take_all()
    # take_all() 返回字典列表，每个字典代表一行
    if parts:
        total_chk = sum(int(p.get("__chk", 0)) for p in parts if isinstance(p, dict))
        return float(total_chk)
    else:
        # 如果没有结果，至少触发执行
        count = chk_ds.count()
        return float(count)


# 向后兼容的别名（保留旧接口，但标记为 deprecated）
def materialize_spark(df: Any) -> None:
    """
    Deprecated: 使用 force_execute() 替代
    
    保留此函数仅用于向后兼容
    """
    import warnings
    warnings.warn(
        "materialize_spark() is deprecated. Use force_execute(df, engine='spark', cols=output_cols) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(df, 'count'):
        df.count()
    else:
        raise TypeError(f"Expected Spark DataFrame, got {type(df)}")


def materialize_ray(ds: Any) -> None:
    """
    Deprecated: 使用 force_execute() 替代
    
    保留此函数仅用于向后兼容
    """
    import warnings
    warnings.warn(
        "materialize_ray() is deprecated. Use force_execute(ds, engine='ray', cols=output_cols) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(ds, 'materialize'):
        materialized = ds.materialize()
        materialized.count()
    elif hasattr(ds, 'count'):
        ds.count()
    else:
        raise TypeError(f"Expected Ray Dataset, got {type(ds)}")
