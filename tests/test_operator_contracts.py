"""
算子契约测试

测试算子实现的正确性和一致性。
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np


def test_standardscaler_contract():
    """测试StandardScaler算子的基本契约"""
    from bench.operator_spec import get_operator_spec

    # 获取算子规格
    spec = get_operator_spec("StandardScaler")

    # 验证规格
    assert spec.name == "StandardScaler"
    assert spec.input_cols == ["x1", "x2"]
    assert spec.output_cols == ["x1_scaled", "x2_scaled"]
    assert spec.params["with_mean"] is True
    assert spec.params["with_std"] is True


def test_standardscaler_ray_implementation():
    """测试Ray StandardScaler实现"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.ray.operators import run_standardscaler

        # 获取规格
        spec = get_operator_spec("StandardScaler")

        # 创建测试数据
        test_df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["A", "B", "C", "A", "B"],
            "text": ["text1", "text2", "text3", "text4", "text5"]
        })

        # 执行算子
        result_df = run_standardscaler(test_df, spec)

        # 验证基本契约
        assert len(result_df) == len(test_df), "输出行数应与输入相同"
        assert "x1_scaled" in result_df.columns, "应包含x1_scaled列"
        assert "x2_scaled" in result_df.columns, "应包含x2_scaled列"

        # 验证标准化结果（均值应接近0，标准差接近1）
        x1_scaled = result_df["x1_scaled"]
        x2_scaled = result_df["x2_scaled"]

        # 检查均值（标准化后应接近0）
        assert abs(x1_scaled.mean()) < 0.1, f"x1_scaled均值不接近0: {x1_scaled.mean()}"
        assert abs(x2_scaled.mean()) < 0.1, f"x2_scaled均值不接近0: {x2_scaled.mean()}"

        # 检查标准差（标准化后应接近1，使用总体标准差ddof=0）
        assert abs(x1_scaled.std(ddof=0) - 1.0) < 0.1, f"x1_scaled标准差不接近1: {x1_scaled.std(ddof=0)}"
        assert abs(x2_scaled.std(ddof=0) - 1.0) < 0.1, f"x2_scaled标准差不接近1: {x2_scaled.std(ddof=0)}"

        # 检查输出列不为空
        assert not x1_scaled.isna().all(), "x1_scaled列全为空"
        assert not x2_scaled.isna().all(), "x2_scaled列全为空"

    except ImportError:
        pytest.skip("Ray依赖未安装")
    except Exception as e:
        pytest.fail(f"Ray StandardScaler测试失败: {e}")


def test_standardscaler_spark_implementation():
    """测试Spark StandardScaler实现"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_standardscaler

        # 获取规格
        spec = get_operator_spec("StandardScaler")

        # 创建Spark会话
        spark = get_spark("TestStandardScaler")

        # 创建测试数据
        test_data = [
            (1.0, 10.0, "A", "text1"),
            (2.0, 20.0, "B", "text2"),
            (3.0, 30.0, "C", "text3"),
            (4.0, 40.0, "A", "text4"),
            (5.0, 50.0, "B", "text5")
        ]
        test_df = spark.createDataFrame(test_data, ["x1", "x2", "cat", "text"])

        # 执行算子
        result_df = run_standardscaler(spark, test_df, spec)

        # 转换为pandas验证
        result_pandas = result_df.toPandas()

        # 验证基本契约
        assert len(result_pandas) == len(test_data), "输出行数应与输入相同"
        assert "x1_scaled" in result_pandas.columns, "应包含x1_scaled列"
        assert "x2_scaled" in result_pandas.columns, "应包含x2_scaled列"

        # 验证标准化结果
        x1_scaled = result_pandas["x1_scaled"]
        x2_scaled = result_pandas["x2_scaled"]

        # 检查均值
        assert abs(x1_scaled.mean()) < 0.1, f"x1_scaled均值不接近0: {x1_scaled.mean()}"
        assert abs(x2_scaled.mean()) < 0.1, f"x2_scaled均值不接近0: {x2_scaled.mean()}"

        # 检查标准差
        assert abs(x1_scaled.std() - 1.0) < 0.1, f"x1_scaled标准差不接近1: {x1_scaled.std()}"
        assert abs(x2_scaled.std() - 1.0) < 0.1, f"x2_scaled标准差不接近1: {x2_scaled.std()}"

        # 检查输出列不为空
        assert not x1_scaled.isna().all(), "x1_scaled列全为空"
        assert not x2_scaled.isna().all(), "x2_scaled列全为空"

        # 清理
        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark StandardScaler测试失败: {e}")


def test_operator_consistency():
    """测试两个引擎的输出一致性"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_standardscaler as spark_scaler
        from engines.ray.operators import run_standardscaler as ray_scaler

        # 获取规格
        spec = get_operator_spec("StandardScaler")

        # 创建相同的测试数据
        test_data = [
            (1.0, 10.0, "A", "text1"),
            (2.0, 20.0, "B", "text2"),
            (3.0, 30.0, "C", "text3"),
            (4.0, 40.0, "A", "text4"),
            (5.0, 50.0, "B", "text5")
        ]

        # Spark执行
        spark = get_spark("ConsistencyTest")
        spark_df = spark.createDataFrame(test_data, ["x1", "x2", "cat", "text"])
        spark_result = run_standardscaler(spark, spark_df, spec)
        spark_pandas = spark_result.toPandas()
        spark.stop()

        # Ray执行
        pandas_df = pd.DataFrame(test_data, columns=["x1", "x2", "cat", "text"])
        ray_result = run_standardscaler(pandas_df, spec)

        # 比较结果结构
        assert list(spark_pandas.columns) == list(ray_result.columns), "列名不一致"
        assert len(spark_pandas) == len(ray_result), "行数不一致"

        # 比较标准化结果（允许小误差）
        tolerance = 0.01
        spark_x1 = spark_pandas["x1_scaled"].values
        ray_x1 = ray_result["x1_scaled"].values
        spark_x2 = spark_pandas["x2_scaled"].values
        ray_x2 = ray_result["x2_scaled"].values

        # 检查数值相似性
        x1_diff = np.abs(spark_x1 - ray_x1)
        x2_diff = np.abs(spark_x2 - ray_x2)

        assert np.all(x1_diff < tolerance), f"x1_scaled差异过大: max={x1_diff.max()}"
        assert np.all(x2_diff < tolerance), f"x2_scaled差异过大: max={x2_diff.max()}"

        print("✓ Spark和Ray的StandardScaler结果一致")

    except ImportError as e:
        pytest.skip(f"依赖未安装: {e}")
    except Exception as e:
        pytest.fail(f"一致性测试失败: {e}")


def test_data_validation():
    """测试数据验证功能"""
    from bench.io import validate_dataframe

    # 创建有效数据
    valid_df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [4.0, 5.0, 6.0],
        "cat": ["A", "B", "C"]
    })

    # 测试有效数据
    errors = validate_dataframe(valid_df, required_cols=["x1", "x2"])
    assert len(errors) == 0, f"有效数据验证失败: {errors}"

    # 测试缺失列
    errors = validate_dataframe(valid_df, required_cols=["x1", "missing_col"])
    assert len(errors) > 0, "应检测到缺失列"

    # 测试空数据
    empty_df = pd.DataFrame()
    errors = validate_dataframe(empty_df)
    assert len(errors) > 0, "应检测到空数据"


def test_metrics_collection():
    """测试指标收集功能"""
    from bench.metrics import MetricsCollector, PerformanceMetrics

    collector = MetricsCollector()

    # 测试计时功能
    import time

    collector.start_measurement()
    time.sleep(0.1)  # 休眠100ms
    elapsed = collector.end_measurement()

    assert elapsed >= 0.1, f"耗时测量不准确: {elapsed}"

    # 测试指标创建
    input_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    output_df = input_df.copy()

    metrics = collector.collect_metrics(input_df, output_df, elapsed)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.wall_time_seconds >= 0.1
    assert metrics.input_rows == 3
    assert metrics.input_cols == 2
    assert metrics.output_rows == 3
    assert metrics.output_cols == 2
    assert metrics.throughput_rows_per_sec > 0


if __name__ == "__main__":
    # 运行所有测试
    print("运行算子契约测试...")

    test_functions = [
        test_standardscaler_contract,
        test_standardscaler_ray_implementation,
        test_standardscaler_spark_implementation,
        test_operator_consistency,
        test_data_validation,
        test_metrics_collection
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"运行 {test_func.__name__}...", end=" ")
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
        print("所有算子契约测试通过！✓")
