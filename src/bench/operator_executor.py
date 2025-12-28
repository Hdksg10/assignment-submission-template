"""
高性能算子执行器工厂

提供最小开销的算子执行机制，支持Spark MLlib和Ray Data算子的自动查找和执行。
"""

import importlib
import time
from typing import Dict, Callable, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd

from .operator_spec import OperatorSpec, get_operator_spec
from .logger import get_logger


@dataclass
class OperatorExecutionContext:
    """算子执行上下文"""
    engine: str
    operator_name: str
    spec: OperatorSpec
    spark_session: Optional[Any] = None
    ray_context: Optional[Any] = None


class HighPerformanceOperatorExecutor:
    """
    高性能算子执行器工厂

    核心优化：
    1. 预加载算子函数，避免每次动态导入
    2. 最小化参数传递开销
    3. 直接返回执行函数指针，减少调用层数
    """

    # 预注册的算子映射 - 避免运行时反射开销
    _OPERATOR_REGISTRY: Dict[str, Dict[str, Callable]] = {}

    # 日志器
    _logger = get_logger(__name__)

    @classmethod
    def register_operator(cls, engine: str, operator_name: str, func: Callable) -> None:
        """预注册算子函数"""
        if engine not in cls._OPERATOR_REGISTRY:
            cls._OPERATOR_REGISTRY[engine] = {}
        cls._OPERATOR_REGISTRY[engine][operator_name] = func

    @classmethod
    def get_operator_func(cls, engine: str, operator_name: str) -> Callable:
        """
        获取算子执行函数 - 零开销查找

        Args:
            engine: 引擎名称 ('spark' 或 'ray')
            operator_name: 算子名称

        Returns:
            算子执行函数

        Raises:
            ValueError: 如果算子未注册
        """
        if engine not in cls._OPERATOR_REGISTRY:
            available_engines = list(cls._OPERATOR_REGISTRY.keys())
            raise ValueError(f"不支持的引擎 '{engine}'，可用引擎: {available_engines}")

        if operator_name not in cls._OPERATOR_REGISTRY[engine]:
            available_ops = list(cls._OPERATOR_REGISTRY[engine].keys())
            raise ValueError(f"引擎 '{engine}' 不支持算子 '{operator_name}'，可用算子: {available_ops}")

        return cls._OPERATOR_REGISTRY[engine][operator_name]

    @classmethod
    def auto_register_operators(cls) -> None:
        """自动注册所有算子 - 在模块导入时调用"""
        # Spark算子
        try:
            from engines.spark.operators import (
                run_standardscaler as spark_standardscaler
            )
            cls.register_operator('spark', 'StandardScaler', spark_standardscaler)
        except ImportError:
            cls._logger.warning("Spark依赖未安装，跳过注册")

        # Ray算子
        try:
            from engines.ray.operators import (
                run_standardscaler_with_ray_data as ray_standardscaler
            )
            cls.register_operator('ray', 'StandardScaler', ray_standardscaler)
        except ImportError:
            cls._logger.warning("Ray依赖未安装，跳过注册")

    @staticmethod
    def create_execution_context(engine: str, operator_name: str, **kwargs) -> OperatorExecutionContext:
        """
        创建执行上下文

        Args:
            engine: 引擎名称
            operator_name: 算子名称
            **kwargs: 额外参数（spark_session, ray_context等）

        Returns:
            OperatorExecutionContext: 执行上下文
        """
        spec = get_operator_spec(operator_name)

        # 记录引擎特定的实现信息（用于调试和日志）
        impl_name = spec.get_engine_impl_name(engine)
        if impl_name:
            HighPerformanceOperatorExecutor._logger.info(f"使用引擎 '{engine}' 的 '{operator_name}' 实现: {impl_name}")

        return OperatorExecutionContext(
            engine=engine,
            operator_name=operator_name,
            spec=spec,
            spark_session=kwargs.get('spark_session'),
            ray_context=kwargs.get('ray_context')
        )


# 在模块导入时自动注册算子
HighPerformanceOperatorExecutor.auto_register_operators()


class DirectOperatorExecutor:
    """
    直接算子执行器 - 最小开销版本

    直接调用算子函数，参数通过位置参数传递，避免字典查找开销。
    """

    @staticmethod
    def execute_spark_operator(operator_func: Callable,
                             input_df,
                             spec: OperatorSpec,
                             spark_session=None) -> Any:
        """
        直接执行Spark算子 - 最小开销

        Args:
            operator_func: 算子函数
            input_df: 输入DataFrame
            spec: 算子规格
            spark_session: Spark会话

        Returns:
            处理后的DataFrame
        """
        return operator_func(spark_session, input_df, spec)

    @staticmethod
    def execute_ray_operator(operator_func: Callable,
                           input_df,
                           spec: OperatorSpec) -> Any:
        """
        直接执行Ray算子 - 最小开销

        Args:
            operator_func: 算子函数
            input_df: 输入DataFrame或Ray Dataset
            spec: 算子规格

        Returns:
            处理后的DataFrame或Ray Dataset
        """
        return operator_func(input_df, spec)

    @staticmethod
    def execute_operator(context: OperatorExecutionContext,
                        input_df) -> Any:
        """
        统一的算子执行接口

        Args:
            context: 执行上下文
            input_df: 输入数据

        Returns:
            处理后的数据
        """
        operator_func = HighPerformanceOperatorExecutor.get_operator_func(
            context.engine, context.operator_name
        )

        if context.engine == 'spark':
            return DirectOperatorExecutor.execute_spark_operator(
                operator_func, input_df, context.spec, context.spark_session
            )
        elif context.engine == 'ray':
            return DirectOperatorExecutor.execute_ray_operator(
                operator_func, input_df, context.spec
            )
        else:
            raise ValueError(f"不支持的引擎: {context.engine}")


class PerformanceOptimizedTimer:
    """
    高精度性能计时器

    使用time.perf_counter()提供纳秒级精度，避免系统时钟调整影响。
    """

    def __init__(self):
        self._logger = get_logger(__name__)
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """开始计时"""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        if self.start_time is None:
            raise RuntimeError("必须先调用start()")

        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    def measure(self, func: Callable, *args, **kwargs) -> tuple:
        """
        测量函数执行时间

        Args:
            func: 要测量的函数
            *args, **kwargs: 函数参数

        Returns:
            (result, elapsed_seconds)
        """
        self.start()
        try:
            result = func(*args, **kwargs)
            elapsed = self.stop()
            return result, elapsed
        except Exception as e:
            # 即使出错也要停止计时
            self.stop()
            self._logger.error(f"函数执行失败: {e}")
            raise e
