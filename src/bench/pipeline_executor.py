"""
高性能管道执行器

支持多个算子顺序执行，自动处理数据流转换，最小化包装开销。
"""

import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd

from .operator_spec import OperatorSpec
from .operator_executor import (
    HighPerformanceOperatorExecutor,
    DirectOperatorExecutor,
    OperatorExecutionContext,
    PerformanceOptimizedTimer
)
from .logger import get_logger


@dataclass
class PipelineStep:
    """管道步骤定义"""
    operator_name: str
    spec: OperatorSpec
    params_override: Optional[Dict[str, Any]] = None

    def get_effective_spec(self) -> OperatorSpec:
        """获取实际使用的规格（合并参数覆盖）"""
        if not self.params_override:
            return self.spec

        # 合并参数
        merged_params = self.spec.params.copy()
        merged_params.update(self.params_override)

        return OperatorSpec(
            name=self.spec.name,
            input_cols=self.spec.input_cols,
            output_cols=self.spec.output_cols,
            params=merged_params,
            description=self.spec.description
        )


@dataclass
class PipelineConfig:
    """管道配置"""
    name: str
    description: str
    steps: List[PipelineStep]
    engine: str = "spark"

    @classmethod
    def from_operator_names(cls, operator_names: List[str],
                          engine: str = "spark",
                          name: str = None) -> 'PipelineConfig':
        """
        从算子名称列表创建管道配置

        Args:
            operator_names: 算子名称列表
            engine: 引擎名称
            name: 管道名称（可选）

        Returns:
            PipelineConfig: 管道配置
        """
        from .operator_spec import get_operator_spec

        steps = []
        current_input_cols = None

        for operator_name in operator_names:
            # 获取原始spec（不要修改注册表中的对象）
            original_spec = get_operator_spec(operator_name)

            # 如果是第一个算子，使用默认输入列
            if current_input_cols is None:
                current_input_cols = original_spec.input_cols

            # 创建新的spec拷贝，不修改原对象
            step_spec = OperatorSpec(
                name=original_spec.name,
                input_cols=current_input_cols,  # 使用推导的输入列
                output_cols=original_spec.output_cols,
                params=original_spec.params.copy(),
                description=original_spec.description
            )

            step = PipelineStep(
                operator_name=operator_name,
                spec=step_spec
            )
            steps.append(step)

            # 下一步的输入列是当前步骤的输出列
            current_input_cols = step_spec.output_cols

        pipeline_name = name or f"pipeline_{'_'.join(operator_names)}"

        return cls(
            name=pipeline_name,
            description=f"算子管道: {' -> '.join(operator_names)}",
            steps=steps,
            engine=engine
        )


class HighPerformancePipelineExecutor:
    """
    高性能管道执行器

    核心优化：
    1. 预创建所有执行上下文，避免运行时开销
    2. 最小化数据拷贝和转换
    3. 支持Spark和Ray Data的原生数据流
    4. 精确的性能测量
    """

    def __init__(self, engine: str, spark_session=None, ray_context=None):
        """
        初始化管道执行器

        Args:
            engine: 引擎名称 ('spark' 或 'ray')
            spark_session: Spark会话（Spark引擎需要）
            ray_context: Ray上下文（Ray引擎需要）
        """
        self.engine = engine
        self.spark_session = spark_session
        self.ray_context = ray_context
        self.timer = PerformanceOptimizedTimer()

    def create_execution_contexts(self, steps: List[PipelineStep]) -> List[OperatorExecutionContext]:
        """
        预创建所有执行上下文

        Args:
            steps: 管道步骤列表

        Returns:
            执行上下文列表
        """
        contexts = []
        for step in steps:
            context = HighPerformanceOperatorExecutor.create_execution_context(
                engine=self.engine,
                operator_name=step.operator_name,
                spark_session=self.spark_session,
                ray_context=self.ray_context
            )
            # 使用步骤的实际规格
            context.spec = step.get_effective_spec()
            contexts.append(context)

        return contexts

    def execute_pipeline(self,
                        steps: List[PipelineStep],
                        input_df,
                        measure_performance: bool = False,
                        per_step_timing: bool = False) -> Union[Any, Tuple[Any, float]]:
        """
        执行算子管道 - 高性能版本

        Args:
            steps: 管道步骤列表
            input_df: 输入DataFrame（pandas或Spark）
            measure_performance: 是否测量性能
            per_step_timing: 是否需要每步计时（会导致每步都触发action）

        Returns:
            如果measure_performance=False: 最终处理后的DataFrame
            如果measure_performance=True: (最终DataFrame, 总耗时)
        """
        if not steps:
            if measure_performance:
                return input_df, 0.0
            return input_df

        # 预创建执行上下文
        contexts = self.create_execution_contexts(steps)

        current_df = input_df

        if measure_performance:
            self.timer.start()

        try:
            # 顺序执行所有算子
            for i, context in enumerate(contexts):
                operator_func = HighPerformanceOperatorExecutor.get_operator_func(
                    context.engine, context.operator_name
                )

                current_df = DirectOperatorExecutor.execute_operator(context, current_df)

                # 只在需要per-step timing时才每步触发action
                # 否则只构建变换链，最后统一触发
                if per_step_timing:
                    # 需要persist/cache避免重复计算
                    if self.engine == 'spark' and hasattr(current_df, 'cache'):
                        current_df = current_df.cache()
                        current_df.count()  # 触发cache
                    elif self.engine == 'ray' and hasattr(current_df, 'materialize'):
                        current_df = current_df.materialize()

            # 在最后触发一次action（如果还没触发）
            if measure_performance and not per_step_timing:
                if self.engine == 'spark' and hasattr(current_df, 'count'):
                    current_df.count()  # 触发整个pipeline的执行
                elif self.engine == 'ray' and hasattr(current_df, 'materialize'):
                    current_df = current_df.materialize()
                    current_df.count()

            if measure_performance:
                total_elapsed = self.timer.stop()
                return current_df, total_elapsed
            else:
                return current_df

        except Exception as e:
            if measure_performance:
                self.timer.stop()
            raise e

    def execute_pipeline_with_detailed_metrics(self,
                                             steps: List[PipelineStep],
                                             input_df) -> Dict[str, Any]:
        """
        执行管道并收集详细性能指标（每步计时）
        
        注意：每步计时会导致每步都触发action，并使用cache/materialize避免重复计算。
        这不是"算子本体"的纯净计时，而是包含了持久化开销。

        Args:
            steps: 管道步骤列表
            input_df: 输入DataFrame

        Returns:
            包含详细性能指标的结果字典
        """
        if not steps:
            return {
                'final_df': input_df,
                'total_time': 0.0,
                'step_times': [],
                'step_details': []
            }

        contexts = self.create_execution_contexts(steps)
        current_df = input_df

        step_times = []
        step_details = []

        total_start = time.perf_counter()

        for i, context in enumerate(contexts):
            step_start = time.perf_counter()

            operator_func = HighPerformanceOperatorExecutor.get_operator_func(
                context.engine, context.operator_name
            )

            current_df = DirectOperatorExecutor.execute_operator(context, current_df)

            # 每步都需要persist/materialize + trigger
            if self.engine == 'spark' and hasattr(current_df, 'cache'):
                current_df = current_df.cache()
                current_df.count()  # 触发cache
            elif self.engine == 'ray' and hasattr(current_df, 'materialize'):
                current_df = current_df.materialize()

            step_elapsed = time.perf_counter() - step_start

            step_times.append(step_elapsed)
            step_details.append({
                'step': i + 1,
                'operator': context.operator_name,
                'time': round(step_elapsed, 4),
                'input_rows': getattr(input_df, 'shape', [0, 0])[0] if i == 0 else None,
                'output_rows': getattr(current_df, 'shape', [0, 0])[0] if hasattr(current_df, 'shape') else None
            })

        total_elapsed = time.perf_counter() - total_start

        return {
            'final_df': current_df,
            'total_time': round(total_elapsed, 4),
            'step_times': [round(t, 4) for t in step_times],
            'step_details': step_details
        }


class OptimizedPipelineRunner:
    """
    优化的管道运行器 - 专门用于性能测试

    提供预热、重复执行等性能测试功能。
    """

    def __init__(self, engine: str, repeats: int = 3, warmup: bool = True):
        """
        初始化管道运行器

        Args:
            engine: 引擎名称
            repeats: 重复执行次数
            warmup: 是否执行预热
        """
        self.engine = engine
        self.repeats = repeats
        self.warmup = warmup
        self.timer = PerformanceOptimizedTimer()
        self._logger = get_logger(__name__)

    def run_pipeline_experiment(self,
                               steps: List[PipelineStep],
                               input_df,
                               spark_session=None) -> Dict[str, Any]:
        """
        运行管道性能实验

        Args:
            steps: 管道步骤
            input_df: 输入数据
            spark_session: Spark会话

        Returns:
            实验结果字典
        """
        executor = HighPerformancePipelineExecutor(
            engine=self.engine,
            spark_session=spark_session
        )

        # 预热运行
        if self.warmup:
            self._logger.info("执行预热运行...")
            try:
                _ = executor.execute_pipeline(steps, input_df, measure_performance=False)
                self._logger.info("预热运行完成")
            except Exception as e:
                self._logger.warning(f"预热运行失败: {e}")

        # 正式运行
        all_times = []
        self._logger.info(f"开始 {self.repeats} 次重复实验...")

        for i in range(self.repeats):
            self._logger.debug(f"运行 {i+1}/{self.repeats}...")

            _, elapsed = executor.execute_pipeline(steps, input_df, measure_performance=True)
            all_times.append(elapsed)
            self._logger.info(f"运行 {i+1} 完成，耗时: {elapsed:.3f}s")

        # 计算统计结果
        import statistics
        avg_time = statistics.mean(all_times)
        std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
        min_time = min(all_times)
        max_time = max(all_times)

        # 计算吞吐量（如果有行数信息）
        throughput = None
        if hasattr(input_df, 'shape'):
            rows = input_df.shape[0]
            throughput = rows / avg_time if avg_time > 0 else 0

        return {
            'engine': self.engine,
            'pipeline_steps': len(steps),
            'repeats': self.repeats,
            'warmup': self.warmup,
            'times': [round(t, 4) for t in all_times],
            'avg_time': round(avg_time, 4),
            'std_time': round(std_time, 4),
            'min_time': round(min_time, 4),
            'max_time': round(max_time, 4),
            'throughput_rows_per_sec': round(throughput, 2) if throughput else None,
            'input_rows': input_df.shape[0] if hasattr(input_df, 'shape') else None,
            'input_cols': input_df.shape[1] if hasattr(input_df, 'shape') else None
        }
