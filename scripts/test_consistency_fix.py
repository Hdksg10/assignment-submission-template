#!/usr/bin/env python3
"""
测试 Ray 和 Spark StandardScaler 的列一致性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

# 测试数据
test_data = [
    (1.0, 10.0, "A", "text1"),
    (2.0, 20.0, "B", "text2"),
    (3.0, 30.0, "C", "text3"),
    (4.0, 40.0, "A", "text4"),
    (5.0, 50.0, "B", "text5")
]

print("=" * 60)
print("测试 StandardScaler 列一致性")
print("=" * 60)

# 获取规格
from bench.operator_spec import get_operator_spec
spec = get_operator_spec("StandardScaler")
print(f"\nOperatorSpec:")
print(f"  input_cols: {spec.input_cols}")
print(f"  output_cols: {spec.output_cols}")

# 测试 Spark
print("\n" + "=" * 60)
print("Spark 实现")
print("=" * 60)
try:
    from engines.spark.session import get_spark
    from engines.spark.operators import run_standardscaler as spark_scaler
    
    spark = get_spark("ConsistencyTest")
    spark_df = spark.createDataFrame(test_data, ["x1", "x2", "cat", "text"])
    spark_result = spark_scaler(spark, spark_df, spec)
    spark_pandas = spark_result.toPandas()
    
    print(f"输出列: {list(spark_pandas.columns)}")
    print(f"列数: {len(spark_pandas.columns)}")
    print(f"行数: {len(spark_pandas)}")
    print("\n前3行数据:")
    print(spark_pandas.head(3))
    
    spark.stop()
except Exception as e:
    print(f"Spark 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 Ray
print("\n" + "=" * 60)
print("Ray 实现")
print("=" * 60)
try:
    from engines.ray.operators import run_standardscaler as ray_scaler
    
    pandas_df = pd.DataFrame(test_data, columns=["x1", "x2", "cat", "text"])
    ray_result_ds = ray_scaler(pandas_df, spec)
    ray_result = ray_result_ds.to_pandas()
    
    print(f"输出列: {list(ray_result.columns)}")
    print(f"列数: {len(ray_result.columns)}")
    print(f"行数: {len(ray_result)}")
    print("\n前3行数据:")
    print(ray_result.head(3))
except Exception as e:
    print(f"Ray 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("一致性检查")
print("=" * 60)
try:
    spark_cols = list(spark_pandas.columns)
    ray_cols = list(ray_result.columns)
    
    if spark_cols == ray_cols:
        print("✓ 列名一致！")
        print(f"  列: {spark_cols}")
    else:
        print("✗ 列名不一致")
        print(f"  Spark: {spark_cols}")
        print(f"  Ray:   {ray_cols}")
        print(f"  Spark 独有: {set(spark_cols) - set(ray_cols)}")
        print(f"  Ray 独有:   {set(ray_cols) - set(spark_cols)}")
except NameError:
    print("✗ 无法比较（某个引擎测试失败）")

