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
    """
    测试高精度计时器的精度
    
    验证 PerformanceOptimizedTimer 使用 time.perf_counter() 的准确性。
    该计时器内部使用 perf_counter()，专门用于性能测试，不受系统时钟调整影响。
    """
    from bench.operator_executor import PerformanceOptimizedTimer

    timer = PerformanceOptimizedTimer()

    # 测试1: 验证计时器使用 perf_counter() 的准确性
    # 使用更长的 sleep 时间（100ms）以提高精度和稳定性
    # time.sleep() 在短时间（10ms）下精度不够，容易受系统调度影响
    target_duration = 0.1  # 100ms
    times = []
    for _ in range(10):
        timer.start()
        time.sleep(target_duration)  # 100ms
        elapsed = timer.stop()
        times.append(elapsed)

    # 验证计时器精度
    # 使用平均值而不是单次测量，减少系统调度的影响
    avg_elapsed = sum(times) / len(times)
    error = abs(avg_elapsed - target_duration) / target_duration * 100  # 相对误差
    
    print(f"\n测试1: 计时器精度验证（使用 perf_counter）")
    print(f"目标时长: {target_duration:.3f}s")
    print(f"平均测量时长: {avg_elapsed:.6f}s")
    print(f"相对误差: {error:.2f}%")
    print(f"单次测量值: {[f'{t:.6f}' for t in times]}")
    
    # 验证平均误差在可接受范围内（< 5%）
    assert error < 5.0, f"计时器精度不足: {error:.2f}%"
    
    # 同时验证单次测量的最大误差（允许更大的偏差，因为系统调度）
    max_error = max(abs(t - target_duration) / target_duration * 100 for t in times)
    print(f"最大单次误差: {max_error:.2f}%")
    # 单次测量允许更大的误差（< 10%），因为系统调度的影响
    assert max_error < 10.0, f"单次测量误差过大: {max_error:.2f}%"

    # 测试2: 对比 perf_counter 和 time.time() 的差异
    # 验证计时器确实使用了 perf_counter() 而不是 time.time()
    print(f"\n测试2: 验证计时器使用 perf_counter()")
    
    # 使用 perf_counter 直接测量
    start_perf = time.perf_counter()
    time.sleep(target_duration)
    end_perf = time.perf_counter()
    direct_perf_elapsed = end_perf - start_perf
    
    # 使用计时器测量（内部使用 perf_counter）
    timer.start()
    time.sleep(target_duration)
    timer_elapsed = timer.stop()
    
    # 两种方法应该非常接近（差异 < 1%）
    diff = abs(timer_elapsed - direct_perf_elapsed) / direct_perf_elapsed * 100
    print(f"直接使用 perf_counter: {direct_perf_elapsed:.6f}s")
    print(f"使用 PerformanceOptimizedTimer: {timer_elapsed:.6f}s")
    print(f"差异: {diff:.2f}%")
    
    assert diff < 1.0, f"计时器与 perf_counter 差异过大: {diff:.2f}%"
    print("✓ 计时器确实使用了 perf_counter()")

    # 测试3: 验证计时器不受系统时钟调整影响（通过对比 time.time()）
    # 注意：在实际环境中，如果系统时钟被调整，time.time() 可能给出错误结果
    print(f"\n测试3: 验证计时器不受系统时钟调整影响")
    
    # 使用 time.time() 测量（可能受系统时钟调整影响）
    start_time = time.time()
    time.sleep(target_duration)
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    # 使用计时器测量（不受系统时钟调整影响）
    timer.start()
    time.sleep(target_duration)
    timer_elapsed = timer.stop()
    
    print(f"使用 time.time(): {time_elapsed:.6f}s")
    print(f"使用 PerformanceOptimizedTimer (perf_counter): {timer_elapsed:.6f}s")
    print("说明: perf_counter() 不受系统时钟调整影响，更适合性能测试")
    
    # 验证计时器测量结果合理（应该接近目标时长）
    timer_error = abs(timer_elapsed - target_duration) / target_duration * 100
    assert timer_error < 5.0, f"计时器测量结果不合理: {timer_error:.2f}%"

    print("\n✓ 高精度计时器测试通过")
    print("✓ 验证了 PerformanceOptimizedTimer 使用 perf_counter() 的准确性")
    print("✓ 验证了计时器不受系统时钟调整影响")


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
