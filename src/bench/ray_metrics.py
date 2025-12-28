"""
Ray特定的性能测试工具

确保Ray Data的lazy execution被正确触发，提供精确的性能测量。
"""

import time
from typing import Callable, Any, Optional, Dict, Union
import pandas as pd

# 延迟导入Ray，避免硬依赖
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

try:
    from .logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    # 如果logger模块不可用，使用print作为fallback
    _logger = None


class RayPerformanceProfiler:
    """
    Ray性能分析器

    专门处理Ray Data的lazy execution特性，确保性能测量准确性。
    """

    def __init__(self):
        if not RAY_AVAILABLE:
            raise ImportError("Ray不可用，请安装Ray: pip install ray")
        self.timer = self._create_high_precision_timer()
        self._logger = _logger or self._create_fallback_logger()

    def _create_high_precision_timer(self) -> Callable[[], float]:
        """创建高精度计时器"""
        return time.perf_counter

    def _create_fallback_logger(self):
        """创建fallback logger"""
        class FallbackLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return FallbackLogger()

    def measure_ray_data_execution(self,
                                  dataset_func: Callable,
                                  trigger_func: Optional[Callable] = None,
                                  **trigger_kwargs) -> tuple:
        """
        精确测量Ray Data执行时间

        Args:
            dataset_func: 返回Ray Dataset的函数
            trigger_func: 触发执行的函数（默认：to_pandas）
            **trigger_kwargs: 传递给触发函数的参数

        Returns:
            (result, elapsed_seconds)
        """
        if trigger_func is None:
            trigger_func = lambda ds: ds.to_pandas()

        # 创建数据集（lazy）
        dataset = dataset_func()

        # 开始计时
        start_time = self.timer()

        try:
            # 触发执行
            result = trigger_func(dataset, **trigger_kwargs)
            elapsed = self.timer() - start_time

            return result, elapsed

        except Exception as e:
            # 即使出错也要计算耗时
            elapsed = self.timer() - start_time
            self._logger.error(f"Ray Data执行失败，耗时: {elapsed:.4f}s")
            raise e

    def measure_ray_task_execution(self,
                                  task_func: Callable,
                                  *args,
                                  **kwargs) -> tuple:
        """
        测量Ray Task执行时间

        Args:
            task_func: Ray任务函数
            *args, **kwargs: 函数参数

        Returns:
            (result, elapsed_seconds)
        """
        start_time = self.timer()

        try:
            result = task_func(*args, **kwargs)

            # 如果返回Ray ObjectRef，等待完成
            if isinstance(result, ray.ObjectRef):
                result = ray.get(result)

            elapsed = self.timer() - start_time
            return result, elapsed

        except Exception as e:
            elapsed = self.timer() - start_time
            self._logger.error(f"Ray Task执行失败，耗时: {elapsed:.4f}s")
            raise e

    def get_ray_cluster_stats(self) -> Dict[str, Any]:
        """获取Ray集群统计信息"""
        if not ray.is_initialized():
            return {"status": "not_initialized"}

        try:
            return {
                "status": "initialized",
                "version": ray.__version__,
                "nodes": len(ray.nodes()),
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "alive_nodes": sum(1 for node in ray.nodes() if node.get("alive", False))
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


class RayDataTrigger:
    """
    Ray Data执行触发器

    提供多种触发Ray Data执行的方式，确保lazy操作被实际执行。
    """

    @staticmethod
    def trigger_to_pandas(dataset, **kwargs) -> pd.DataFrame:
        """触发执行并转换为pandas DataFrame"""
        return dataset.to_pandas(**kwargs)

    @staticmethod
    def trigger_count(dataset, **kwargs) -> int:
        """触发执行并返回行数"""
        return dataset.count()

    @staticmethod
    def trigger_collect(dataset, **kwargs) -> list:
        """触发执行并收集结果"""
        return dataset.take_all()

    @staticmethod
    def trigger_with_custom_func(dataset, func: Callable, **kwargs):
        """使用自定义函数触发执行"""
        return func(dataset, **kwargs)


class RayOptimizedExperimentRunner:
    """
    针对Ray优化的实验运行器

    专门处理Ray Data的性能测试，确保测量准确性。
    """

    def __init__(self, repeats: int = 3, warmup: bool = True):
        """
        初始化Ray实验运行器

        Args:
            repeats: 重复执行次数
            warmup: 是否预热
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray不可用，请安装Ray: pip install ray")
        self.repeats = repeats
        self.warmup = warmup
        self.profiler = RayPerformanceProfiler()
        self._logger = _logger or self._create_fallback_logger()

    def _create_fallback_logger(self):
        """创建fallback logger"""
        class FallbackLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return FallbackLogger()

    def run_ray_data_experiment(self,
                               dataset_func: Callable,
                               trigger_func: Optional[Callable] = None,
                               **trigger_kwargs) -> Dict[str, Any]:
        """
        运行Ray Data性能实验

        Args:
            dataset_func: 创建Ray Dataset的函数
            trigger_func: 触发执行的函数
            **trigger_kwargs: 触发函数参数

        Returns:
            实验结果字典
        """
        if trigger_func is None:
            trigger_func = RayDataTrigger.trigger_to_pandas

        # 预热运行
        if self.warmup:
            self._logger.info("执行Ray Data预热运行...")
            try:
                _, _ = self.profiler.measure_ray_data_execution(
                    dataset_func, trigger_func, **trigger_kwargs
                )
                self._logger.info("预热运行完成")
            except Exception as e:
                self._logger.warning(f"预热运行失败: {e}")

        # 正式运行
        all_times = []
        results = []

        self._logger.info(f"开始 {self.repeats} 次Ray Data重复实验...")

        for i in range(self.repeats):
            self._logger.debug(f"运行 {i+1}/{self.repeats}...")

            try:
                result, elapsed = self.profiler.measure_ray_data_execution(
                    dataset_func, trigger_func, **trigger_kwargs
                )

                all_times.append(elapsed)
                results.append(result)

                self._logger.info(f"运行 {i+1} 完成，耗时: {elapsed:.3f}s")

            except Exception as e:
                self._logger.error(f"运行 {i+1} 失败: {e}")
                all_times.append(-1)  # 失败标记
                results.append(None)

        # 计算统计结果
        import statistics
        valid_times = [t for t in all_times if t > 0]

        if not valid_times:
            return {
                'engine': 'ray',
                'experiment_type': 'ray_data',
                'success': False,
                'error': '所有运行都失败了'
            }

        avg_time = statistics.mean(valid_times)
        std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
        min_time = min(valid_times)
        max_time = max(valid_times)

        # 计算吞吐量（如果结果是DataFrame）
        throughput = None
        input_rows = None
        if results and isinstance(results[0], pd.DataFrame):
            input_rows = len(results[0])
            throughput = input_rows / avg_time if avg_time > 0 else 0

        # 获取集群统计信息
        cluster_stats = self.profiler.get_ray_cluster_stats()

        return {
            'engine': 'ray',
            'experiment_type': 'ray_data',
            'success': True,
            'repeats': self.repeats,
            'warmup': self.warmup,
            'times': [round(t, 4) for t in all_times],
            'valid_times': [round(t, 4) for t in valid_times],
            'avg_time': round(avg_time, 4),
            'std_time': round(std_time, 4),
            'min_time': round(min_time, 4),
            'max_time': round(max_time, 4),
            'throughput_rows_per_sec': round(throughput, 2) if throughput else None,
            'input_rows': input_rows,
            'cluster_stats': cluster_stats
        }


def create_ray_data_experiment_func(dataset_func: Callable,
                                  operator_func: Callable,
                                  *operator_args,
                                  **operator_kwargs) -> Callable:
    """
    创建Ray Data实验函数

    Args:
        dataset_func: 创建基础数据集的函数
        operator_func: 算子执行函数
        *operator_args, **operator_kwargs: 算子参数

    Returns:
        实验函数（返回Ray Dataset）
    """
    def experiment_func():
        # 创建基础数据集
        dataset = dataset_func()

        # 应用算子
        result_dataset = operator_func(dataset, *operator_args, **operator_kwargs)

        return result_dataset

    return experiment_func
