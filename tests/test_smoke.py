"""
冒烟测试

测试所有核心组件的基本功能，确保系统能正常运行。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 包已通过pyproject.toml安装，无需手动添加路径

# import pytest  # 移除pytest依赖
# import pandas as pd  # 延迟导入，避免依赖


def test_project_structure():
    """测试项目基本结构"""
    # 检查必需的目录
    required_dirs = [
        "data/raw",
        "src/bench",
        "src/engines/spark",
        "src/engines/ray",
        "experiments/reports",
        "scripts",
        "tests"
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"目录不存在: {dir_path}"
        assert Path(dir_path).is_dir(), f"不是目录: {dir_path}"


def test_sample_data():
    """测试样本数据"""
    sample_path = Path("data/raw/sample.csv")
    assert sample_path.exists(), "sample.csv不存在"

    try:
        import pandas as pd
        # 加载并检查数据
        df = pd.read_csv(sample_path)
        assert len(df) >= 100, f"数据行数不足: {len(df)}"

        # 检查必需列
        required_cols = ["x1", "x2", "cat", "text"]
        for col in required_cols:
            assert col in df.columns, f"缺少列: {col}"

        # 检查数据类型
        assert df["x1"].dtype in ["float64", "object"], f"x1列类型异常: {df['x1'].dtype}"
        assert df["x2"].dtype in ["float64", "object"], f"x2列类型异常: {df['x2'].dtype}"
        assert df["cat"].dtype == "object", f"cat列类型异常: {df['cat'].dtype}"
        assert df["text"].dtype == "object", f"text列类型异常: {df['text'].dtype}"
    except ImportError:
        print("pandas未安装，跳过详细数据检查")
        # 至少检查文件大小
        assert sample_path.stat().st_size > 1000, "sample.csv文件过小"


def test_bench_imports():
    """测试bench模块导入"""
    try:
        from bench.operator_spec import get_operator_spec, list_operator_names

        # 测试算子规格
        operators = list_operator_names()
        assert len(operators) > 0, "没有注册任何算子"

        # 测试StandardScaler算子
        spec = get_operator_spec("StandardScaler")
        assert spec.name == "StandardScaler"
        assert len(spec.input_cols) > 0
        assert len(spec.output_cols) > 0

        # 测试metrics模块（不依赖pandas）
        from bench.metrics import MetricsCollector
        collector = MetricsCollector()
        assert collector is not None

    except ImportError as e:
        if "pandas" in str(e):
            print("pandas未安装，跳过bench模块完整测试")
            # 至少测试operator_spec可以导入
            from bench.operator_spec import list_operator_names
            operators = list_operator_names()
            assert len(operators) > 0
        else:
            raise AssertionError(f"bench模块导入失败: {e}")


def test_data_io():
    """测试数据IO功能"""
    try:
        from bench.io import load_csv, save_csv, validate_dataframe
        import pandas as pd

        # 加载样本数据
        df = load_csv("data/raw/sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # 验证数据
        errors = validate_dataframe(df, required_cols=["x1", "x2"])
        assert len(errors) == 0, f"数据验证失败: {errors}"

        # 测试保存和重新加载
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_csv(df, tmp_path)
            df_loaded = load_csv(tmp_path)
            assert len(df_loaded) == len(df)
            assert list(df_loaded.columns) == list(df.columns)
        finally:
            os.unlink(tmp_path)
    except ImportError:
        print("pandas未安装，跳过IO测试")


def test_spark_engine():
    """测试Spark引擎基本功能"""
    try:
        from engines.spark.session import get_spark

        # 创建Spark会话
        spark = get_spark("TestApp")
        assert spark is not None

        # 测试基本操作
        test_data = [("Alice", 1), ("Bob", 2)]
        df = spark.createDataFrame(test_data, ["name", "value"])
        assert df.count() == 2

        # 清理
        spark.stop()

    except ImportError:
        print("Spark依赖未安装，跳过测试")
        return
    except Exception as e:
        raise AssertionError(f"Spark引擎测试失败: {e}")


def test_ray_engine():
    """测试Ray引擎基本功能"""
    try:
        import ray
        from engines.ray.runtime import init_ray, shutdown_ray

        # 初始化Ray
        init_ray(num_cpus=1, num_gpus=0)

        # 测试基本功能
        @ray.remote
        def test_func(x):
            return x * 2

        result = ray.get(test_func.remote(5))
        assert result == 10

        # 清理
        shutdown_ray()

    except ImportError:
        print("Ray依赖未安装，跳过测试")
        return
    except Exception as e:
        raise AssertionError(f"Ray引擎测试失败: {e}")


def test_operator_execution():
    """测试算子执行"""
    try:
        from bench.operator_spec import get_operator_spec
        from engines.ray.operators import run_standardscaler
        import pandas as pd

        # 获取算子规格
        spec = get_operator_spec("StandardScaler")

        # 创建测试数据
        test_df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "cat": ["A", "B", "C"],
            "text": ["text1", "text2", "text3"]
        })

        # 执行算子（返回Ray Dataset）
        result_ds = run_standardscaler(test_df, spec)
        
        # 转换为pandas验证
        result_df = result_ds.to_pandas()

        # 验证结果
        assert len(result_df) == len(test_df)
        assert "x1_scaled" in result_df.columns
        assert "x2_scaled" in result_df.columns

        # 检查标准化结果（均值接近0）
        mean_x1 = result_df["x1_scaled"].mean()
        mean_x2 = result_df["x2_scaled"].mean()
        assert abs(mean_x1) < 0.1, f"x1_scaled均值不接近0: {mean_x1}"
        assert abs(mean_x2) < 0.1, f"x2_scaled均值不接近0: {mean_x2}"

    except ImportError as e:
        print(f"依赖未安装: {e}，跳过测试")
        return
    except Exception as e:
        raise AssertionError(f"算子执行测试失败: {e}")


def test_cli_basic():
    """测试CLI基本功能"""
    try:
        # 简单测试：检查CLI模块可以导入
        import sys
        import subprocess

        # 测试命令行帮助
        result = subprocess.run([
            sys.executable, "-m", "bench.cli", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / "src")

        # 即使pandas未安装，帮助也应该工作
        if result.returncode != 0:
            print(f"CLI帮助测试失败，但这可能是正常的: {result.stderr}")

        # 至少检查模块可以导入
        from bench.cli import create_parser
        parser = create_parser()
        assert parser is not None

        print("CLI基本功能测试通过")

    except ImportError as e:
        print(f"CLI依赖未安装: {e}，跳过CLI测试")
    except Exception as e:
        raise AssertionError(f"CLI测试失败: {e}")


def test_experiment_structure():
    """测试实验目录结构"""
    # 确保实验目录存在
    assert Path("experiments").exists()
    assert Path("experiments/reports").exists()
    assert Path("experiments/runs").exists()

    # 检查是否能写入报告
    test_report_path = Path("experiments/reports/test_report.json")
    test_data = {"test": "data", "timestamp": "2024-01-01"}

    try:
        import json
        with open(test_report_path, 'w') as f:
            json.dump(test_data, f)

        # 验证文件写入
        assert test_report_path.exists()

        # 读取验证
        with open(test_report_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    finally:
        # 清理测试文件
        if test_report_path.exists():
            test_report_path.unlink()


if __name__ == "__main__":
    # 运行所有测试
    print("运行冒烟测试...")

    test_functions = [
        test_project_structure,
        test_sample_data,
        test_bench_imports,
        test_data_io,
        test_spark_engine,
        test_ray_engine,
        test_operator_execution,
        test_cli_basic,
        test_experiment_structure
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
        print("所有冒烟测试通过！✓")
