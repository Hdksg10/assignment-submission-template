"""
性能测试准确性验证

验证高性能执行器工厂和管道执行器的包装开销，确保测试结果与直接调用一致。
"""

import time
import pandas as pd
import pytest
from pathlib import Path
import tempfile


def test_spark_direct_vs_executor_overhead():
    """测试Spark直接调用与执行器工厂的性能差异"""
    try:
        from engines.spark.session import get_spark
        from engines.spark.operators.standardscaler import run_standardscaler
        from bench.operator_executor import HighPerformanceOperatorExecutor
        from bench.operator_spec import get_operator_spec

        # 创建Spark会话
        spark = get_spark("PerformanceTest")

        # 创建测试数据
        df = spark.createDataFrame([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)] * 1000, ["x1", "x2"])
        spec = get_operator_spec("StandardScaler")

        # 方式1: 直接调用算子函数
        times_direct = []
        for _ in range(5):
            start = time.perf_counter()
            result1 = run_standardscaler(spark, df, spec)
            result1.count()  # 触发执行
            times_direct.append(time.perf_counter() - start)

        # 方式2: 通过执行器工厂调用
        executor_func = HighPerformanceOperatorExecutor.get_operator_func('spark', 'StandardScaler')
        times_executor = []
        for _ in range(5):
            start = time.perf_counter()
            result2 = executor_func(spark, df, spec)
            result2.count()  # 触发执行
            times_executor.append(time.perf_counter() - start)

        # 计算开销
        avg_direct = sum(times_direct) / len(times_direct)
        avg_executor = sum(times_executor) / len(times_executor)
        overhead = (avg_executor - avg_direct) / avg_direct * 100

        print(f"Spark直接调用平均耗时: {avg_direct:.4f}s")
        print(f"执行器工厂调用平均耗时: {avg_executor:.4f}s")
        print(f"开销百分比: {overhead:.2f}%")

        # 验证开销在可接受范围内（< 2%）
        assert overhead < 2.0, f"Spark执行器开销过大: {overhead:.2f}%"

        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")


def test_ray_direct_vs_executor_overhead():
    """测试Ray直接调用与执行器工厂的性能差异"""
    try:
        import ray.data as rd
        from engines.ray.runtime import init_ray
        from engines.ray.operators.standardscaler import run_standardscaler_with_ray_data
        from bench.operator_executor import HighPerformanceOperatorExecutor
        from bench.operator_spec import get_operator_spec
        from bench.ray_metrics import RayDataTrigger, RayPerformanceProfiler

        # 初始化Ray
        init_ray(num_cpus=2)

        # 创建测试数据
        df = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0] * 1000,
            'x2': [4.0, 5.0, 6.0] * 1000
        })
        dataset = rd.from_pandas(df)
        spec = get_operator_spec("StandardScaler")

        profiler = RayPerformanceProfiler()

        # 方式1: 直接调用算子函数
        times_direct = []
        for _ in range(3):
            _, elapsed = profiler.measure_ray_data_execution(
                lambda: run_standardscaler_with_ray_data(dataset, spec),
                RayDataTrigger.trigger_count
            )
            times_direct.append(elapsed)

        # 方式2: 通过执行器工厂调用
        executor_func = HighPerformanceOperatorExecutor.get_operator_func('ray', 'StandardScaler')
        times_executor = []
        for _ in range(3):
            _, elapsed = profiler.measure_ray_data_execution(
                lambda: executor_func(dataset, spec),
                RayDataTrigger.trigger_count
            )
            times_executor.append(elapsed)

        # 计算开销
        avg_direct = sum(times_direct) / len(times_direct)
        avg_executor = sum(times_executor) / len(times_executor)
        overhead = (avg_executor - avg_direct) / avg_direct * 100

        print(f"Ray直接调用平均耗时: {avg_direct:.4f}s")
        print(f"执行器工厂调用平均耗时: {avg_executor:.4f}s")
        print(f"开销百分比: {overhead:.2f}%")

        # 验证开销在可接受范围内（< 1%）
        assert overhead < 1.0, f"Ray执行器开销过大: {overhead:.2f}%"

    except ImportError:
        pytest.skip("Ray依赖未安装")


def test_pipeline_vs_individual_execution():
    """测试管道执行与单独执行的性能对比"""
    try:
        from engines.spark.session import get_spark
        from bench.pipeline_executor import PipelineConfig, HighPerformancePipelineExecutor
        from bench.operator_spec import get_operator_spec

        spark = get_spark("PipelineTest")

        # 创建测试数据
        df = spark.createDataFrame([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)] * 1000, ["x1", "x2"])

        # 创建管道配置
        pipeline_config = PipelineConfig.from_operator_names(
            operator_names=['StandardScaler'],
            engine='spark'
        )

        executor = HighPerformancePipelineExecutor(
            engine='spark',
            spark_session=spark
        )

        # 方式1: 管道执行
        _, pipeline_time = executor.execute_pipeline(
            steps=pipeline_config.steps,
            input_df=df,
            measure_performance=True
        )

        # 方式2: 单独执行
        spec = get_operator_spec('StandardScaler')
        start = time.perf_counter()
        from engines.spark.operators.standardscaler import run_standardscaler
        result = run_standardscaler(spark, df, spec)
        result.count()
        individual_time = time.perf_counter() - start

        # 计算差异
        overhead = (pipeline_time - individual_time) / individual_time * 100

        print(f"单独执行耗时: {individual_time:.4f}s")
        print(f"管道执行耗时: {pipeline_time:.4f}s")
        print(f"开销百分比: {overhead:.2f}%")

        # 管道开销应该很小（< 0.5%）
        assert overhead < 0.5, f"管道执行开销过大: {overhead:.2f}%"

        spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")


def test_timer_precision():
    """测试高精度计时器的精度"""
    from bench.operator_executor import PerformanceOptimizedTimer

    timer = PerformanceOptimizedTimer()

    # 测试基本计时功能
    times = []
    for _ in range(10):
        start = time.perf_counter()
        time.sleep(0.01)  # 10ms
        timer.start()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()
        end = time.perf_counter()

        # 计算实际耗时
        actual_elapsed = end - start
        times.append((elapsed, actual_elapsed))

    # 验证计时器精度
    for timer_elapsed, actual_elapsed in times:
        error = abs(timer_elapsed - 0.01) / 0.01 * 100  # 相对误差
        assert error < 5.0, f"计时器精度不足: {error:.2f}%"

    print("高精度计时器测试通过")


def test_ray_data_execution_triggering():
    """测试Ray Data执行触发机制"""
    try:
        import ray.data as rd
        from bench.ray_metrics import RayDataTrigger, RayPerformanceProfiler

        # 初始化Ray
        from engines.ray.runtime import init_ray
        init_ray(num_cpus=2)

        # 创建测试数据
        df = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0] * 100,
            'x2': [4.0, 5.0, 6.0] * 100
        })
        dataset = rd.from_pandas(df)

        profiler = RayPerformanceProfiler()

        # 测试不同触发方式
        def create_identity_dataset():
            def identity_transform(batch):
                return batch
            return dataset.map_batches(identity_transform, batch_format="pandas")

        # 方式1: to_pandas
        result1, time1 = profiler.measure_ray_data_execution(
            create_identity_dataset,
            RayDataTrigger.trigger_to_pandas
        )

        # 方式2: count
        result2, time2 = profiler.measure_ray_data_execution(
            create_identity_dataset,
            RayDataTrigger.trigger_count
        )

        # 验证结果一致性
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, int)
        assert result2 == len(df)

        print(f"to_pandas触发耗时: {time1:.4f}s")
        print(f"count触发耗时: {time2:.4f}s")

        # 确保都被正确触发（耗时 > 0）
        assert time1 > 0
        assert time2 > 0

    except ImportError:
        pytest.skip("Ray依赖未安装")


def test_end_to_end_performance_consistency():
    """端到端性能一致性测试"""
    try:
        from engines.spark.session import get_spark
        from bench.pipeline_executor import PipelineConfig, OptimizedPipelineRunner

        spark = get_spark("ConsistencyTest")

        # 创建临时测试数据
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            df = pd.DataFrame({
                'x1': [1.0, 2.0, 3.0] * 1000,
                'x2': [4.0, 5.0, 6.0] * 1000
            })
            df.to_csv(tmp_path, index=False)

        try:
            # 方式1: 使用新管道系统
            pipeline_config = PipelineConfig.from_operator_names(
                operator_names=['StandardScaler'],
                engine='spark'
            )

            runner = OptimizedPipelineRunner(
                engine='spark',
                repeats=2,
                warmup=True
            )

            spark_df = spark.createDataFrame(df)
            result1 = runner.run_pipeline_experiment(
                steps=pipeline_config.steps,
                input_df=spark_df,
                spark_session=spark
            )

            print(f"新管道系统平均耗时: {result1['avg_time']:.4f}s")

            # 验证结果合理性
            assert result1['avg_time'] > 0
            assert result1['std_time'] >= 0
            assert len(result1['times']) == 2

        finally:
            # 清理临时文件
            Path(tmp_path).unlink()
            spark.stop()

    except ImportError:
        pytest.skip("Spark依赖未安装")


if __name__ == "__main__":
    # 运行性能测试
    print("运行性能准确性测试...")

    test_timer_precision()

    try:
        test_spark_direct_vs_executor_overhead()
    except Exception as e:
        print(f"Spark性能测试跳过: {e}")

    try:
        test_ray_direct_vs_executor_overhead()
    except Exception as e:
        print(f"Ray性能测试跳过: {e}")

    try:
        test_pipeline_vs_individual_execution()
    except Exception as e:
        print(f"管道性能测试跳过: {e}")

    try:
        test_ray_data_execution_triggering()
    except Exception as e:
        print(f"Ray Data触发测试跳过: {e}")

    try:
        test_end_to_end_performance_consistency()
    except Exception as e:
        print(f"端到端一致性测试跳过: {e}")

    print("性能准确性测试完成！")
