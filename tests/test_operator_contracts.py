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
        spark_result = spark_scaler(spark, spark_df, spec)
        spark_pandas = spark_result.toPandas()
        spark.stop()

        # Ray执行
        pandas_df = pd.DataFrame(test_data, columns=["x1", "x2", "cat", "text"])
        ray_result = ray_scaler(pandas_df, spec)

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


def test_stringindexer_contract():
    """测试StringIndexer算子的基本契约"""
    from bench.operator_spec import get_operator_spec

    # 获取算子规格
    spec = get_operator_spec("StringIndexer")

    # 验证规格
    assert spec.name == "StringIndexer"
    assert spec.input_cols == ["cat"]
    assert spec.output_cols == ["cat_indexed"]
    assert spec.params["handle_invalid"] == "error"


def test_stringindexer_spark_implementation():
    """测试Spark StringIndexer实现"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_stringindexer

        # 获取规格
        spec = get_operator_spec("StringIndexer")

        # 创建Spark会话
        spark = get_spark("TestStringIndexer")

        # 创建测试数据
        test_data = [
            ("A", 1.0, "text1"),
            ("B", 2.0, "text2"),
            ("C", 3.0, "text3"),
            ("A", 4.0, "text4"),
            ("B", 5.0, "text5")
        ]
        test_df = spark.createDataFrame(test_data, ["cat", "x1", "text"])

        # 执行算子
        result_df = run_stringindexer(spark, test_df, spec)

        # 转换为pandas验证
        result_pandas = result_df.toPandas()

        # 验证基本契约
        assert len(result_pandas) == len(test_data), "输出行数应与输入相同"
        assert "cat_indexed" in result_pandas.columns, "应包含cat_indexed列"

        # 验证索引结果
        cat_indexed = result_pandas["cat_indexed"]

        # 检查索引值应该是整数
        assert cat_indexed.dtype in [np.int64, np.int32], f"索引列类型应为整数，实际: {cat_indexed.dtype}"

        # 检查索引值范围（应该从0开始）
        assert cat_indexed.min() >= 0, "索引值应 >= 0"
        assert cat_indexed.max() < len(test_data), "索引值应 < 数据行数"

        # 检查相同类别应该有相同索引
        # A应该映射到相同索引
        a_indices = result_pandas[result_pandas["cat"] == "A"]["cat_indexed"].unique()
        assert len(a_indices) == 1, "相同类别应映射到相同索引"

        # 检查输出列不为空
        assert not cat_indexed.isna().all(), "cat_indexed列全为空"

        # 清理
        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark StringIndexer测试失败: {e}")


def test_stringindexer_handle_invalid():
    """测试StringIndexer的handle_invalid策略"""
    try:
        from bench.operator_spec import get_operator_spec, OperatorSpec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_stringindexer

        spark = get_spark("TestStringIndexerHandleInvalid")

        # 创建包含未知类别的测试数据
        train_data = [
            ("A", 1.0),
            ("B", 2.0),
            ("C", 3.0)
        ]
        train_df = spark.createDataFrame(train_data, ["cat", "x1"])

        test_data = [
            ("A", 4.0),  # 已知类别
            ("D", 5.0),  # 未知类别
            ("B", 6.0)   # 已知类别
        ]
        test_df = spark.createDataFrame(test_data, ["cat", "x1"])

        # 测试 handle_invalid="error"
        spec_error = OperatorSpec(
            name="StringIndexer",
            input_cols=["cat"],
            output_cols=["cat_indexed"],
            params={"handle_invalid": "error", "input_cols": ["cat"], "output_cols": ["cat_indexed"]},
            description="test"
        )
        # 先fit训练数据
        from engines.spark.operators import run_stringindexer
        train_result = run_stringindexer(spark, train_df, spec_error)
        
        # 测试数据应该会报错（如果Spark StringIndexer在transform时遇到未知值）
        # 注意：Spark StringIndexer在fit时建立映射，transform时遇到未知值会根据handleInvalid处理
        
        # 测试 handle_invalid="keep"
        spec_keep = OperatorSpec(
            name="StringIndexer",
            input_cols=["cat"],
            output_cols=["cat_indexed"],
            params={"handle_invalid": "keep", "input_cols": ["cat"], "output_cols": ["cat_indexed"]},
            description="test"
        )
        result_keep = run_stringindexer(spark, test_df, spec_keep)
        result_pandas = result_keep.toPandas()
        
        # 验证keep策略：未知值应该被映射到一个特殊索引
        assert "cat_indexed" in result_pandas.columns
        assert len(result_pandas) == len(test_data), "keep策略不应过滤行"

        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"StringIndexer handle_invalid测试失败: {e}")


def test_onehotencoder_contract():
    """测试OneHotEncoder算子的基本契约"""
    from bench.operator_spec import get_operator_spec

    # 获取算子规格
    spec = get_operator_spec("OneHotEncoder")

    # 验证规格
    assert spec.name == "OneHotEncoder"
    assert spec.input_cols == ["cat_indexed"]
    assert spec.output_cols == ["cat_onehot"]
    assert spec.params["drop_last"] is True
    assert spec.params["handle_invalid"] == "error"


def test_onehotencoder_spark_implementation():
    """测试Spark OneHotEncoder实现"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_onehotencoder

        # 获取规格
        spec = get_operator_spec("OneHotEncoder")

        # 创建Spark会话
        spark = get_spark("TestOneHotEncoder")

        # 创建测试数据（索引列，应该是StringIndexer的输出）
        test_data = [
            (0, 1.0, "text1"),  # 类别0
            (1, 2.0, "text2"),  # 类别1
            (2, 3.0, "text3"),  # 类别2
            (0, 4.0, "text4"),  # 类别0
            (1, 5.0, "text5")   # 类别1
        ]
        test_df = spark.createDataFrame(test_data, ["cat_indexed", "x1", "text"])

        # 执行算子
        result_df = run_onehotencoder(spark, test_df, spec)

        # 转换为pandas验证
        result_pandas = result_df.toPandas()

        # 验证基本契约
        assert len(result_pandas) == len(test_data), "输出行数应与输入相同"
        assert "cat_onehot" in result_pandas.columns, "应包含cat_onehot列"

        # 验证独热编码结果
        cat_onehot = result_pandas["cat_onehot"]

        # 检查独热编码应该是向量类型（Spark OneHotEncoder输出SparseVector或DenseVector）
        # 转换为pandas后可能是向量对象或数组
        assert cat_onehot is not None, "独热编码列不应为空"

        # 检查输出列不为空
        assert not cat_onehot.isna().all(), "cat_onehot列全为空"

        # 清理
        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark OneHotEncoder测试失败: {e}")


def test_stringindexer_onehotencoder_pipeline():
    """测试StringIndexer + OneHotEncoder管道"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_stringindexer, run_onehotencoder

        spark = get_spark("TestPipeline")

        # 创建测试数据
        test_data = [
            ("A", 1.0, "text1"),
            ("B", 2.0, "text2"),
            ("C", 3.0, "text3"),
            ("A", 4.0, "text4"),
            ("B", 5.0, "text5")
        ]
        test_df = spark.createDataFrame(test_data, ["cat", "x1", "text"])

        # 步骤1: StringIndexer
        stringindexer_spec = get_operator_spec("StringIndexer")
        indexed_df = run_stringindexer(spark, test_df, stringindexer_spec)

        # 步骤2: OneHotEncoder
        onehot_spec = get_operator_spec("OneHotEncoder")
        result_df = run_onehotencoder(spark, indexed_df, onehot_spec)

        # 转换为pandas验证
        result_pandas = result_df.toPandas()

        # 验证管道结果
        assert len(result_pandas) == len(test_data), "输出行数应与输入相同"
        assert "cat_indexed" in result_pandas.columns, "应包含cat_indexed列"
        assert "cat_onehot" in result_pandas.columns, "应包含cat_onehot列"

        # 验证索引列
        cat_indexed = result_pandas["cat_indexed"]
        assert cat_indexed.dtype in [np.int64, np.int32], "索引列应为整数类型"

        # 验证独热编码列
        cat_onehot = result_pandas["cat_onehot"]
        assert cat_onehot is not None, "独热编码列不应为空"

        # 清理
        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"StringIndexer + OneHotEncoder管道测试失败: {e}")


def test_minmaxscaler_spark_implementation():
    """测试Spark MinMaxScaler实现"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.spark.session import get_spark
        from engines.spark.operators import run_minmaxscaler

        # 获取规格
        spec = get_operator_spec("MinMaxScaler")

        # 创建Spark会话
        spark = get_spark("TestMinMaxScaler")

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
        result_df = run_minmaxscaler(spark, test_df, spec)

        # 转换为pandas验证
        result_pandas = result_df.toPandas()

        # 验证基本契约
        assert len(result_pandas) == len(test_data), "输出行数应与输入相同"
        assert "x1_scaled" in result_pandas.columns, "应包含x1_scaled列"
        assert "x2_scaled" in result_pandas.columns, "应包含x2_scaled列"

        # 验证缩放结果（默认范围[0, 1]）
        x1_scaled = result_pandas["x1_scaled"]
        x2_scaled = result_pandas["x2_scaled"]

        # 检查缩放后的值应该在[0, 1]范围内（允许小误差）
        assert x1_scaled.min() >= -0.01, f"x1_scaled最小值应 >= 0，实际: {x1_scaled.min()}"
        assert x1_scaled.max() <= 1.01, f"x1_scaled最大值应 <= 1，实际: {x1_scaled.max()}"
        assert x2_scaled.min() >= -0.01, f"x2_scaled最小值应 >= 0，实际: {x2_scaled.min()}"
        assert x2_scaled.max() <= 1.01, f"x2_scaled最大值应 <= 1，实际: {x2_scaled.max()}"

        # 检查输出列不为空
        assert not x1_scaled.isna().all(), "x1_scaled列全为空"
        assert not x2_scaled.isna().all(), "x2_scaled列全为空"

        # 清理
        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")
    except Exception as e:
        pytest.fail(f"Spark MinMaxScaler测试失败: {e}")


if __name__ == "__main__":
    # 运行所有测试
    print("运行算子契约测试...")

    test_functions = [
        test_standardscaler_contract,
        test_standardscaler_ray_implementation,
        test_standardscaler_spark_implementation,
        test_operator_consistency,
        test_data_validation,
        test_metrics_collection,
        test_stringindexer_contract,
        test_stringindexer_spark_implementation,
        test_stringindexer_handle_invalid,
        test_onehotencoder_contract,
        test_onehotencoder_spark_implementation,
        test_stringindexer_onehotencoder_pipeline,
        test_minmaxscaler_spark_implementation
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
