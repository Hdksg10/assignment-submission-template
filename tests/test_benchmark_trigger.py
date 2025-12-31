"""
防回归测试：确保 Spark 触发不会被 column pruning 裁剪

测试目的：防止未来又用回 count() 造成"测不到 transform"的系统性偏差
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np


def test_spark_force_execute_depends_on_output_values():
    """
    测试 Spark force_execute 依赖输出列的值（防止 column pruning）
    
    通过验证不同变换得到不同 checksum，证明 force_execute 确实依赖输出列值
    """
    try:
        from engines.spark.session import get_spark
        from bench.materialize import force_execute
        from pyspark.sql import functions as F
        
        spark = get_spark("TriggerTest")
        
        try:
            # 创建测试数据
            df = spark.createDataFrame([(1.0,), (3.0,), (5.0,)], ["x"])
            
            # 构造两种不同的 z 变换
            df2 = df.withColumn("z", F.col("x") * 2.0)
            df3 = df.withColumn("z", F.col("x") * 3.0)
            
            # 使用 force_execute 触发，依赖输出列 z 的值
            chk2 = force_execute(df2, engine="spark", cols=["z"])
            chk3 = force_execute(df3, engine="spark", cols=["z"])
            
            # 关键断言：checksum 必须不同，证明 force_execute 依赖输出列值
            assert chk2 != chk3, \
                "force_execute 必须依赖输出列 z 的值，否则无法防 column pruning"
            
            # 对照：count 不依赖列值，两者应相同
            assert df2.count() == df3.count() == 3, \
                "count() 不依赖列值，两种变换的行数应该相同"
            
        finally:
            spark.stop()
            
    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark force_execute 防回归测试失败: {e}")


def test_spark_force_execute_with_multiple_output_cols():
    """
    测试 Spark force_execute 处理多个输出列
    """
    try:
        from engines.spark.session import get_spark
        from bench.materialize import force_execute
        from pyspark.sql import functions as F
        
        spark = get_spark("TriggerTestMulti")
        
        try:
            # 创建测试数据
            test_data = [(1.0, 2.0), (3.0, 4.0)]
            df = spark.createDataFrame(test_data, ["x", "y"])
            
            # 创建两种不同的多列变换
            df_a = df.withColumn("z1", F.col("x") * 2.0).withColumn("z2", F.col("y") * 3.0)
            df_b = df.withColumn("z1", F.col("x") * 2.0).withColumn("z2", F.col("y") * 4.0)
            
            # 使用 force_execute 触发，依赖多个输出列
            chk_a = force_execute(df_a, engine="spark", cols=["z1", "z2"])
            chk_b = force_execute(df_b, engine="spark", cols=["z1", "z2"])
            
            # 关键断言：checksum 必须不同，证明依赖输出列值
            assert chk_a != chk_b, \
                "force_execute 必须依赖输出列 z1, z2 的值"
            
        finally:
            spark.stop()
            
    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark force_execute 多列测试失败: {e}")


def test_ray_force_execute_does_not_call_materialize(monkeypatch):
    """
    测试 Ray force_execute 不调用 materialize（避免物化开销）
    
    使用 monkeypatch 确保 materialize() 被调用时会报错
    """
    try:
        import ray
        import ray.data as rd
        from ray.data.dataset import Dataset
        from bench.materialize import force_execute
        import pandas as pd
        
        ray.init(ignore_reinit_error=True)
        
        try:
            # Patch materialize 方法，如果被调用就报错
            def _boom(*args, **kwargs):
                raise AssertionError("force_execute 不应调用 Dataset.materialize()")
            
            monkeypatch.setattr(Dataset, "materialize", _boom, raising=True)
            
            # 创建测试数据并做变换
            ds = rd.from_pandas(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
            
            def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
                df["z"] = df["x"] * 2.0
                return df
            
            ds2 = ds.map_batches(transform_batch, batch_format="pandas")
            
            # 只要这里不抛 boom，就证明没调用 materialize
            _ = force_execute(ds2, engine="ray", cols=["z"])
            
        finally:
            ray.shutdown()
            
    except ImportError:
        pytest.skip("Ray依赖未安装")
    except Exception as e:
        pytest.fail(f"Ray force_execute materialize 测试失败: {e}")


def test_ray_force_execute_depends_on_output_values():
    """
    测试 Ray force_execute 依赖输出列的值（触发真实计算）
    """
    try:
        import ray
        import ray.data as rd
        from bench.materialize import force_execute
        import pandas as pd
        
        ray.init(ignore_reinit_error=True)
        
        try:
            # 创建测试数据
            test_data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            ds = rd.from_pandas(test_data)
            
            # 构造两种不同的 z 变换
            def transform_batch_2x(df: pd.DataFrame) -> pd.DataFrame:
                df["z"] = df["x"] * 2.0
                return df
            
            def transform_batch_3x(df: pd.DataFrame) -> pd.DataFrame:
                df["z"] = df["x"] * 3.0
                return df
            
            ds2 = ds.map_batches(transform_batch_2x, batch_format="pandas")
            ds3 = ds.map_batches(transform_batch_3x, batch_format="pandas")
            
            # 使用 force_execute 触发，依赖输出列 z 的值
            chk2 = force_execute(ds2, engine="ray", cols=["z"])
            chk3 = force_execute(ds3, engine="ray", cols=["z"])
            
            # 关键断言：checksum 必须不同，证明 force_execute 依赖输出列值
            assert chk2 != chk3, \
                "force_execute 必须依赖输出列 z 的值，否则无法触发真实计算"
            
        finally:
            ray.shutdown()
            
    except ImportError:
        pytest.skip("Ray依赖未安装")
    except Exception as e:
        pytest.fail(f"Ray force_execute 值依赖测试失败: {e}")


def test_force_execute_fallback_to_count():
    """
    测试 force_execute 在输出列为空时的 fallback 行为
    """
    try:
        from engines.spark.session import get_spark
        from bench.materialize import force_execute
        
        spark = get_spark("TriggerTestFallback")
        
        try:
            # 创建测试数据
            test_data = [(1.0,), (2.0,)]
            df = spark.createDataFrame(test_data, ["x"])
            
            # 使用空的输出列列表（应该 fallback 到 count）
            # 这不应该 crash
            chk = force_execute(df, engine="spark", cols=[])
            
            # 验证返回了行数（count 的结果）
            assert chk == 2.0, "空 cols 时应 fallback 到 count，返回行数"
            
        finally:
            spark.stop()
            
    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"force_execute fallback 测试失败: {e}")


if __name__ == "__main__":
    # 运行所有测试
    print("运行 benchmark trigger 防回归测试...")
    
    test_functions = [
        test_spark_force_execute_depends_on_output_values,
        test_spark_force_execute_with_multiple_output_cols,
        test_ray_force_execute_does_not_call_materialize,
        test_ray_force_execute_depends_on_output_values,
        test_force_execute_fallback_to_count,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"运行 {test_func.__name__}...", end=" ")
            # 对于需要 monkeypatch 的测试，需要特殊处理
            if test_func.__name__ == "test_ray_force_execute_does_not_call_materialize":
                # 在非 pytest 环境下，跳过需要 monkeypatch 的测试
                print("跳过（需要 pytest monkeypatch）")
                continue
            test_func()
            print("✓ 通过")
            passed += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            failed += 1
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("所有防回归测试通过！✓")
