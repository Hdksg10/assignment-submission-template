"""
指标采集模块

提供性能指标的收集、计算和报告功能。
"""

import time
import statistics
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
import json
import subprocess

from .logger import get_logger


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    wall_time_seconds: float
    throughput_rows_per_sec: float
    input_rows: int
    input_cols: int
    output_rows: int
    output_cols: int
    timestamp: str
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_id: str
    engine: str
    operator: str
    dataset_path: str
    git_commit: Optional[str]
    metrics: List[PerformanceMetrics]
    avg_wall_time: float
    std_wall_time: float
    avg_throughput: float
    std_throughput: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['metrics'] = [m.to_dict() for m in self.metrics]
        return result


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start_measurement(self) -> None:
        """开始测量"""
        self.start_time = time.time()

    def end_measurement(self) -> float:
        """结束测量并返回耗时"""
        if self.start_time is None:
            raise RuntimeError("必须先调用start_measurement()")

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        return elapsed

    def collect_metrics(self,
                       input_rows: int,
                       input_cols: int,
                       output_rows: int,
                       output_cols: int,
                       elapsed_seconds: float,
                       memory_mb: Optional[float] = None) -> PerformanceMetrics:
        """
        收集完整的性能指标

        Args:
            input_rows: 输入行数
            input_cols: 输入列数
            output_rows: 输出行数
            output_cols: 输出列数
            elapsed_seconds: 耗时（秒）
            memory_mb: 内存使用（MB），可选

        Returns:
            PerformanceMetrics: 性能指标
        """
        # 计算吞吐量
        throughput = input_rows / elapsed_seconds if elapsed_seconds > 0 else 0

        metrics = PerformanceMetrics(
            wall_time_seconds=round(elapsed_seconds, 3),
            throughput_rows_per_sec=round(throughput, 2),
            input_rows=input_rows,
            input_cols=input_cols,
            output_rows=output_rows,
            output_cols=output_cols,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            memory_mb=memory_mb
        )

        return metrics


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, repeats: int = 3, warmup: bool = True):
        self.repeats = repeats
        self.warmup = warmup
        self.collector = MetricsCollector()
        self._logger = get_logger(__name__)

    def run_experiment(self,
                      engine: str,
                      operator: str,
                      dataset_path: str,
                      operator_func,
                      input_profile_df: pd.DataFrame,
                      materialize_func,
                      *args,
                      **kwargs) -> ExperimentResult:
        """
        运行完整实验（包含多次重复）

        Args:
            engine: 引擎名称 ('spark' 或 'ray')
            operator: 算子名称
            dataset_path: 数据集路径
            operator_func: 算子执行函数
            input_profile_df: 用于获取输入数据行列数的pandas DataFrame（不参与算子执行）
            materialize_func: 触发分布式执行的函数（在计时区间内调用）
            *args, **kwargs: 传递给算子函数的参数

        Returns:
            ExperimentResult: 实验结果
        """
        experiment_id = self._generate_experiment_id(engine, operator)
        git_commit = self._get_git_commit()

        all_metrics = []

        # 从profile DataFrame获取输入数据的行列数
        input_rows = input_profile_df.shape[0]
        input_cols = input_profile_df.shape[1]

        # Warm-up run - 必须包含算子执行和materialize
        if self.warmup:
            self._logger.info("执行预热运行...")
            try:
                warmup_output = operator_func(*args, **kwargs)
                materialize_func(warmup_output)
                self._logger.info("预热运行完成")
            except Exception as e:
                self._logger.warning(f"预热运行失败: {e}")

        # 正式运行
        self._logger.info(f"开始 {self.repeats} 次重复实验...")
        for i in range(self.repeats):
            self._logger.debug(f"运行 {i+1}/{self.repeats}...")

            # 重新加载数据确保一致性
            fresh_args = args  # 在实际使用中可能需要重新加载数据

            self.collector.start_measurement()

            try:
                output_df = operator_func(*fresh_args, **kwargs)
                # 在计时区间内触发分布式执行
                materialize_func(output_df)
                elapsed = self.collector.end_measurement()

                # 推断输出行列数
                # 对于大多数预处理算子，行数不变
                output_rows = input_rows
                # 列数保守估计：如果有spec，使用spec的output_cols长度
                spec = kwargs.get('spec')
                if spec and hasattr(spec, 'output_cols'):
                    output_cols = input_cols + len(spec.output_cols) - len(spec.input_cols)
                else:
                    output_cols = input_cols

                # 收集指标
                metrics = self.collector.collect_metrics(
                    input_rows=input_rows,
                    input_cols=input_cols,
                    output_rows=output_rows,
                    output_cols=output_cols,
                    elapsed_seconds=elapsed
                )
                all_metrics.append(metrics)
                self._logger.info(f"运行 {i+1} 完成，耗时: {elapsed:.3f}s")
            except Exception as e:
                self._logger.error(f"运行 {i+1} 失败: {e}")
                # 记录失败的指标
                failed_metrics = PerformanceMetrics(
                    wall_time_seconds=-1,
                    throughput_rows_per_sec=0,
                    input_rows=input_rows,
                    input_cols=input_cols,
                    output_rows=0,
                    output_cols=0,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
                )
                all_metrics.append(failed_metrics)

        # 计算统计结果
        wall_times = [m.wall_time_seconds for m in all_metrics if m.wall_time_seconds > 0]
        throughputs = [m.throughput_rows_per_sec for m in all_metrics if m.throughput_rows_per_sec > 0]

        avg_wall_time = statistics.mean(wall_times) if wall_times else 0
        std_wall_time = statistics.stdev(wall_times) if len(wall_times) > 1 else 0
        avg_throughput = statistics.mean(throughputs) if throughputs else 0
        std_throughput = statistics.stdev(throughputs) if len(throughputs) > 1 else 0

        result = ExperimentResult(
            experiment_id=experiment_id,
            engine=engine,
            operator=operator,
            dataset_path=dataset_path,
            git_commit=git_commit,
            metrics=all_metrics,
            avg_wall_time=round(avg_wall_time, 3),
            std_wall_time=round(std_wall_time, 3),
            avg_throughput=round(avg_throughput, 2),
            std_throughput=round(std_throughput, 2)
        )

        return result

    def _generate_experiment_id(self, engine: str, operator: str) -> str:
        """生成实验ID"""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        return f"{engine}_{operator}_{timestamp}"

    def _get_git_commit(self) -> Optional[str]:
        """获取当前Git commit"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # 只取前8位
        except Exception:
            pass
        return None


def save_experiment_result(result: ExperimentResult,
                          output_path: Union[str, Path]) -> None:
    """
    保存实验结果到文件

    Args:
        result: 实验结果
        output_path: 输出路径
    """
    logger = get_logger(__name__)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"实验结果已保存到: {output_path}")


def load_experiment_result(input_path: Union[str, Path]) -> ExperimentResult:
    """
    从文件加载实验结果

    Args:
        input_path: 输入路径

    Returns:
        ExperimentResult: 实验结果
    """
    input_path = Path(input_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 重新构造对象
    metrics = [PerformanceMetrics(**m) for m in data['metrics']]
    data['metrics'] = metrics

    result = ExperimentResult(**data)
    return result
