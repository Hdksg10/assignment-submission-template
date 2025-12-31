#!/usr/bin/env python3
"""调试一致性测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

test_data = [
    (1.0, 10.0, "A", "text1"),
    (2.0, 20.0, "B", "text2"),
    (3.0, 30.0, "C", "text3"),
    (4.0, 40.0, "A", "text4"),
    (5.0, 50.0, "B", "text5")
]

print("=" * 60)
print("测试 StandardScaler 一致性")
print("=" * 60)

# 获取规格
from bench.operator_spec import get_operator_spec
spec = get_operator_spec("StandardScaler")
print(f"\nSpec:")
print(f"  input_cols: {spec.input_cols}")
print(f"  output_cols: {spec.output_cols}")

# 测试 Spark
print("\n" + "=" * 60)
print("Spark 输出")
print("=" * 60)
try:
    from engines.spark.session import get_spark
    from engines.spark.operators import run_standardscaler as spark_scaler
    
    spark = get_spark("DebugTest")
    spark_df = spark.createDataFrame(test_data, ["x1", "x2", "cat", "text"])
    spark_result = spark_scaler(spark, spark_df, spec)
    spark_pandas = spark_result.toPandas()
    
    print(f"列: {list(spark_pandas.columns)}")
    print(f"列数: {len(spark_pandas.columns)}")
    print(f"\n前3行:")
    print(spark_pandas.head(3))
    
    spark.stop()
except Exception as e:
    print(f"Spark 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 Ray
print("\n" + "=" * 60)
print("Ray 输出")
print("=" * 60)
try:
    from engines.ray.operators import run_standardscaler as ray_scaler
    
    pandas_df = pd.DataFrame(test_data, columns=["x1", "x2", "cat", "text"])
    ray_result_ds = ray_scaler(pandas_df, spec)
    ray_result = ray_result_ds.to_pandas()
    
    print(f"列: {list(ray_result.columns)}")
    print(f"列数: {len(ray_result.columns)}")
    print(f"\n前3行:")
    print(ray_result.head(3))
except Exception as e:
    print(f"Ray 失败: {e}")
    import traceback
    traceback.print_exc()

# 比较
print("\n" + "=" * 60)
print("比较")
print("=" * 60)
try:
    spark_cols = list(spark_pandas.columns)
    ray_cols = list(ray_result.columns)
    
    print(f"Spark 列: {spark_cols}")
    print(f"Ray 列:   {ray_cols}")
    
    if spark_cols == ray_cols:
        print("✓ 列名完全一致")
    else:
        print("✗ 列名不一致")
        if set(spark_cols) == set(ray_cols):
            print("  但列集合相同（只是顺序不同）")
            print(f"  Spark 顺序: {spark_cols}")
            print(f"  Ray 顺序:   {ray_cols}")
except NameError:
    print("无法比较（某个引擎失败）")

