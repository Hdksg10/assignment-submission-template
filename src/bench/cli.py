"""
命令行接口

提供统一的实验运行和对比接口。
"""

import argparse
import sys
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json

# 导入项目模块
from .operator_spec import get_operator_spec, list_operator_names
from .io import load_csv, save_csv, get_file_info
from .metrics import ExperimentRunner, save_experiment_result
from .logger import get_logger, setup_logging
from .data_ingest import load_input_for_engine, load_input_pandas
from .operator_executor import HighPerformanceOperatorExecutor
# 导入引擎算子模块以确保注册代码执行
try:
    from ..engines.spark.operators import *  # 触发 __init__.py 中的注册
except ImportError:
    pass
try:
    from ..engines.ray.operators import *  # 触发 __init__.py 中的注册
except ImportError:
    pass
import logging


def _sanitize_spark_app_name(name: str, max_len: int = 80) -> str:
    """
    清理 Spark 应用名称，仅允许 [0-9A-Za-z._-] 字符
    
    Args:
        name: 原始应用名称
        max_len: 最大长度限制（默认 80）
    
    Returns:
        清理后的应用名称，如果为空则返回 "BenchmarkApp"
    """
    if not name:
        return "BenchmarkApp"
    
    # 替换不允许的字符为下划线
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    # 合并连续的下划线
    cleaned = re.sub(r"_+", "_", cleaned)
    # 去除首尾下划线
    cleaned = cleaned.strip("_")
    # 截断到最大长度
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    # 如果清理后为空，返回默认值
    if not cleaned:
        return "BenchmarkApp"
    
    return cleaned


def _spark_app_name_for_run(operator_name: str) -> str:
    """
    为 run 命令生成 Spark 应用名称
    
    Args:
        operator_name: 算子名称
    
    Returns:
        清理后的应用名称，格式为 BenchmarkApp-{operator_name}
    """
    return _sanitize_spark_app_name(f"BenchmarkApp-{operator_name}")


def _is_distributed_mode(args) -> bool:
    """
    检测是否为分布式模式（多机集群）
    
    Args:
        args: 命令行参数对象
    
    Returns:
        True 如果是分布式模式，False 否则
    """
    # 检查 Spark master
    spark_master = getattr(args, 'spark_master', None)
    if spark_master and not spark_master.startswith('local'):
        return True
    
    # 检查 Ray address
    ray_address = getattr(args, 'ray_address', None)
    if ray_address is not None:
        return True
    
    return False


def _auto_select_io_mode(args) -> str:
    """
    根据集群模式自动选择 IO 模式
    
    Args:
        args: 命令行参数对象
    
    Returns:
        IO 模式 ('pandas' 或 'engine')
    """
    # 如果用户明确指定了 io_mode，使用用户指定的值
    if hasattr(args, 'io_mode') and args.io_mode:
        return args.io_mode
    
    # 自动检测：如果是分布式模式，默认使用 engine 模式
    if _is_distributed_mode(args):
        return 'engine'
    else:
        return 'pandas'


def parse_spark_conf(conf_list: list) -> dict:
    """
    解析 --spark-conf 参数列表（格式: KEY=VALUE）
    
    Args:
        conf_list: 配置字符串列表，例如 ['spark.executor.memory=4g', 'spark.executor.cores=2']
    
    Returns:
        dict: 配置字典
    """
    config = {}
    for conf_str in conf_list:
        if '=' not in conf_str:
            raise ValueError(f"无效的 spark-conf 格式: {conf_str}。应为 KEY=VALUE")
        key, value = conf_str.split('=', 1)  # 只分割第一个 =
        config[key.strip()] = value.strip()
    return config


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    """
    为子命令添加分布式集群参数
    
    Args:
        parser: 子命令的参数解析器
    """
    # Spark 相关参数
    spark_group = parser.add_argument_group('Spark 集群配置')
    spark_group.add_argument('--spark-master',
                            default=None,
                            help='Spark Master URL (默认: None，使用 local[*]；多机示例: spark://<master-host>:7077)')
    spark_group.add_argument('--spark-driver-host',
                            default=None,
                            help='Spark Driver 主机地址（多机 client mode 强烈建议手动指定 driver 可达 IP）')
    spark_group.add_argument('--spark-conf',
                            action='append',
                            default=[],
                            metavar='KEY=VALUE',
                            help='Spark 配置参数（可重复使用，例如: --spark-conf spark.executor.memory=4g）')

    # Ray 相关参数
    ray_group = parser.add_argument_group('Ray 集群配置')
    ray_group.add_argument('--ray-address',
                          default=None,
                          help='Ray 集群地址 (默认: None，本地模式；多机示例: auto 或 <head-ip>:6379)')
    ray_group.add_argument('--ray-namespace',
                          default='benchmark',
                          help='Ray 命名空间 (默认: benchmark)')
    ray_group.add_argument('--ray-runtime-env-json',
                          default=None,
                          help='Ray 运行时环境 JSON 字符串（用于打包代码依赖等）')

    # 数据读取模式
    io_group = parser.add_argument_group('数据读取配置')
    io_group.add_argument('--io-mode',
                         choices=['pandas', 'engine'],
                         default=None,
                         help='数据读取模式 (默认: 自动选择；多机模式自动使用 engine，单机模式使用 pandas)')
    io_group.add_argument('--data-path',
                         default=None,
                         help='数据文件路径（可选：直接给路径，覆盖 --input；多机时必须是所有节点可访问的共享存储路径）')


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Spark MLlib vs Ray 预处理算子对比工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行单次基准测试
  python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv

  # 运行对比测试
  python -m src.bench.cli compare --operator StandardScaler --input data/raw/sample.csv

  # 运行高性能管道（推荐）
  python -m src.bench.cli pipeline --engine spark --operators StandardScaler --input data/raw/sample.csv

  # 多机 Spark 集群
  python -m src.bench.cli run --engine spark --operator StandardScaler --input hdfs:///data/sample.csv \\
    --spark-master spark://master:7077 --spark-driver-host <driver-ip> --io-mode engine

  # 多机 Ray 集群
  python -m src.bench.cli run --engine ray --operator StandardScaler --input s3://bucket/data/sample.csv \\
    --ray-address <head-ip>:6379 --io-mode engine

  # 查看可用算子
  python -m src.bench.cli list

  # 指定日志等级
  python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv --log-level DEBUG
        """
    )

    # 添加全局日志等级参数
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='设置日志等级 (默认: INFO)')
    parser.add_argument('--py4j-log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='WARNING',
                       help='设置Py4J通信日志等级 (默认: WARNING，减少Spark通信日志)')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='运行单次基准测试')
    run_parser.add_argument('--engine', required=True,
                           choices=['spark', 'ray'],
                           help='使用的引擎')
    run_parser.add_argument('--operator', required=True,
                           help='算子名称')
    run_parser.add_argument('--input', required=True,
                           help='输入数据文件路径')
    run_parser.add_argument('--output', default='experiments/runs/',
                           help='输出目录路径 (默认: experiments/runs/)')
    run_parser.add_argument('--repeats', type=int, default=3,
                           help='重复运行次数 (默认: 3)')
    run_parser.add_argument('--warmup', action='store_true', default=True,
                           help='是否执行预热运行 (默认: True)')
    run_parser.add_argument('--params', type=json.loads, default={},
                           help='额外的算子参数 (JSON格式)')
    run_parser.add_argument('--full-compute', action='store_true', default=False,
                           help='使用全量数据进行 profile 而非小样本（默认: False，使用 limit(1000) 小样本）')
    add_distributed_args(run_parser)

    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='运行对比测试')
    compare_parser.add_argument('--operator', required=True,
                               help='算子名称')
    compare_parser.add_argument('--input', required=True,
                               help='输入数据文件路径')
    compare_parser.add_argument('--output', default='experiments/reports',
                               help='输出目录路径 (默认: experiments/reports)')
    compare_parser.add_argument('--repeats', type=int, default=3,
                               help='重复运行次数 (默认: 3)')
    compare_parser.add_argument('--full-compute', action='store_true', default=False,
                               help='使用全量数据进行 profile 而非小样本（默认: False，使用 limit(1000) 小样本）')
    add_distributed_args(compare_parser)

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出可用算子和引擎')

    # pipeline 命令 - 高性能管道执行
    pipeline_parser = subparsers.add_parser('pipeline', help='运行高性能算子管道')
    pipeline_parser.add_argument('--engine', required=True,
                                choices=['spark', 'ray'],
                                help='使用的引擎')
    pipeline_parser.add_argument('--operators', required=True,
                                nargs='+',
                                help='算子名称列表，按顺序执行')
    pipeline_parser.add_argument('--input', required=True,
                                help='输入数据文件路径')
    pipeline_parser.add_argument('--output', default='experiments/runs/',
                                help='输出目录路径 (默认: experiments/runs/)')
    pipeline_parser.add_argument('--repeats', type=int, default=3,
                                help='重复运行次数 (默认: 3)')
    pipeline_parser.add_argument('--warmup', action='store_true', default=True,
                                help='是否执行预热运行 (默认: True)')
    pipeline_parser.add_argument('--params', type=json.loads, default={},
                                help='算子参数 (JSON格式，key为算子名)')
    pipeline_parser.add_argument('--full-compute', action='store_true', default=False,
                                help='使用全量数据进行 profile 而非小样本（默认: False，使用 limit(1000) 小样本）')
    add_distributed_args(pipeline_parser)

    return parser


def run_single_experiment(args) -> None:
    """运行单次实验"""
    logger = get_logger(__name__)
    try:
        # 获取算子规格
        spec = get_operator_spec(args.operator)
        logger.info(f"运行算子: {spec.name}")
        logger.info(f"描述: {spec.description}")

        # 确定数据路径（优先使用 --data-path，否则使用 --input）
        data_path = getattr(args, 'data_path', None) or args.input
        
        # 自动选择 IO 模式（如果用户未指定）
        io_mode = _auto_select_io_mode(args)
        is_distributed = _is_distributed_mode(args)
        
        # 如果自动选择了 engine 模式，记录日志
        if not hasattr(args, 'io_mode') or not args.io_mode:
            logger.info(f"自动选择 IO 模式: {io_mode} (分布式模式: {is_distributed})")
        
        # 检查多机模式警告（如果用户明确指定了 pandas 模式）
        if is_distributed and io_mode == 'pandas':
            logger.warning(
                "⚠️  多机模式检测到 io-mode=pandas。"
                "pandas->createDataFrame 会让 driver 先读全量，且输入文件在 worker 不可见。"
                "建议使用 --io-mode engine 并使用共享存储路径（NFS/HDFS/S3）。"
            )

        # 合并和验证参数
        operator_params = spec.params.copy()
        if args.params:
            # 验证传入的参数
            for param_name, param_value in args.params.items():
                logger.info(f"使用自定义参数: {param_name} = {param_value}")
                # 可以在这里添加更详细的参数验证逻辑
                if not isinstance(param_name, str):
                    raise ValueError(f"参数名必须是字符串: {param_name}")
            operator_params.update(args.params)
        else:
            logger.info("使用默认参数配置")

        # 更新spec.params，使算子函数能够访问到用户传入的参数
        spec.params = operator_params

        # 动态导入引擎模块
        if args.engine == 'spark':
            from ..engines.spark.session import get_spark

            # 解析 Spark 配置
            spark_config = {}
            if hasattr(args, 'spark_conf') and args.spark_conf:
                spark_config = parse_spark_conf(args.spark_conf)

            # 生成并清理 Spark 应用名称
            spark_app_name = _spark_app_name_for_run(spec.name)
            logger.info(f"Spark app name: {spark_app_name}")

            # 初始化Spark（计时外）
            spark = get_spark(
                app_name=spark_app_name,
                master=getattr(args, 'spark_master', None),
                config=spark_config if spark_config else None,
                driver_host=getattr(args, 'spark_driver_host', None)
            )
            logger.info("Spark会话已初始化")

            # 根据 io-mode 选择数据加载方式
            if io_mode == 'engine':
                # 引擎原生读取（多机推荐）
                spark_df = load_input_for_engine('spark', data_path, spark=spark, is_distributed=is_distributed)
                # 为了 profile，根据 --full-compute 选项决定是否使用全量数据
                full_compute = getattr(args, 'full_compute', False)
                if full_compute:
                    logger.info("使用引擎原生读取，加载全量数据用于 profile")
                    df = spark_df.toPandas()
                else:
                    logger.info("使用引擎原生读取，加载小样本用于 profile")
                    df = spark_df.limit(1000).toPandas() if spark_df.count() > 1000 else spark_df.toPandas()
            else:
                # pandas 模式（单机）
                df = load_input_pandas(data_path)
                spark_df = spark.createDataFrame(df)
                logger.info("数据已转换为Spark DataFrame")

            # 获取算子执行函数
            operator_func = HighPerformanceOperatorExecutor.get_operator_func(args.engine, args.operator)
            logger.info(f"已获取算子函数: {args.operator}")

            # 运行算子
            runner = ExperimentRunner(repeats=args.repeats, warmup=args.warmup)
            result = runner.run_experiment(
                engine=args.engine,
                operator=args.operator,
                dataset_path=args.input,
                operator_func=operator_func,
                input_profile_df=df,  # pandas用于profile
                spark=spark,
                input_df=spark_df,  # Spark DF作为算子输入
                spec=spec
            )

        elif args.engine == 'ray':
            import ray
            import ray.data as rd
            from ..engines.ray.runtime import init_ray

            # 初始化Ray（计时外）
            init_ray(
                address=getattr(args, 'ray_address', None),
                namespace=getattr(args, 'ray_namespace', 'benchmark'),
                runtime_env_json=getattr(args, 'ray_runtime_env_json', None)
            )
            logger.info("Ray运行时已初始化")

            # 根据 io-mode 选择数据加载方式
            if io_mode == 'engine':
                # 引擎原生读取（多机推荐）
                ray_ds = load_input_for_engine('ray', data_path, spark=None, is_distributed=is_distributed)
                # 为了 profile，根据 --full-compute 选项决定是否使用全量数据
                full_compute = getattr(args, 'full_compute', False)
                if full_compute:
                    logger.info("使用引擎原生读取，加载全量数据用于 profile")
                    df = ray_ds.to_pandas()
                else:
                    logger.info("使用引擎原生读取，加载小样本用于 profile")
                    df = ray_ds.limit(1000).to_pandas() if ray_ds.count() > 1000 else ray_ds.to_pandas()
            else:
                # pandas 模式（单机）
                df = load_input_pandas(data_path)
                ray_ds = rd.from_pandas(df)
                logger.info("数据已转换为Ray Dataset")

            # 获取算子执行函数
            operator_func = HighPerformanceOperatorExecutor.get_operator_func(args.engine, args.operator)
            logger.info(f"已获取算子函数: {args.operator}")

            # 运行算子
            runner = ExperimentRunner(repeats=args.repeats, warmup=args.warmup)
            result = runner.run_experiment(
                engine=args.engine,
                operator=args.operator,
                dataset_path=args.input,
                operator_func=operator_func,
                input_profile_df=df,  # pandas用于profile
                ray_dataset=ray_ds,  # Ray Dataset作为算子输入（参数名必须匹配函数签名）
                spec=spec
            )

        # 保存结果
        output_path = Path(args.output) / f"{result.experiment_id}.json"
        save_experiment_result(result, output_path)

        # 打印摘要
        logger.info("实验结果摘要:")
        logger.info(f"引擎: {args.engine}")
        logger.info(f"算子: {args.operator}")
        logger.info(f"平均耗时: {result.avg_wall_time:.3f}s")
        logger.info(f"标准差: {result.std_wall_time:.3f}s")
        logger.info(f"平均吞吐量: {result.avg_throughput:.2f} rows/s")
        logger.info(f"Git Commit: {result.git_commit or 'N/A'}")

    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_comparison_experiment(args) -> None:
    """运行对比实验"""
    logger = get_logger(__name__)
    try:
        logger.info(f"开始对比测试: {args.operator}")

        # 创建临时参数对象用于Spark（继承分布式参数）
        spark_args = argparse.Namespace()
        spark_args.engine = 'spark'
        spark_args.operator = args.operator
        spark_args.input = args.input
        spark_args.output = args.output
        spark_args.repeats = args.repeats
        spark_args.warmup = True
        spark_args.params = {}
        # 继承分布式参数
        spark_args.spark_master = getattr(args, 'spark_master', None)
        spark_args.spark_driver_host = getattr(args, 'spark_driver_host', None)
        spark_args.spark_conf = getattr(args, 'spark_conf', [])
        spark_args.io_mode = getattr(args, 'io_mode', None)  # None 表示自动选择
        spark_args.data_path = getattr(args, 'data_path', None)
        spark_args.full_compute = getattr(args, 'full_compute', False)

        # 创建临时参数对象用于Ray（继承分布式参数）
        ray_args = argparse.Namespace()
        ray_args.engine = 'ray'
        ray_args.operator = args.operator
        ray_args.input = args.input
        ray_args.output = args.output
        ray_args.repeats = args.repeats
        ray_args.warmup = True
        ray_args.params = {}
        # 继承分布式参数
        ray_args.ray_address = getattr(args, 'ray_address', None)
        ray_args.ray_namespace = getattr(args, 'ray_namespace', 'benchmark')
        ray_args.ray_runtime_env_json = getattr(args, 'ray_runtime_env_json', None)
        ray_args.io_mode = getattr(args, 'io_mode', None)  # None 表示自动选择
        ray_args.data_path = getattr(args, 'data_path', None)
        ray_args.full_compute = getattr(args, 'full_compute', False)

        # 运行Spark实验
        logger.info("运行Spark实验")
        run_single_experiment(spark_args)

        # 运行Ray实验
        logger.info("运行Ray实验")
        run_single_experiment(ray_args)

        # 生成对比报告
        logger.info("生成对比报告")
        generate_comparison_report(args.operator, args.input, args.output)

    except Exception as e:
        logger.error(f"对比实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_comparison_report(operator: str, input_path: str, output_dir: str) -> None:
    """生成对比报告"""
    logger = get_logger(__name__)
    try:
        output_path = Path(output_dir)
        report_path = output_path / f"comparison_{operator}_{Path(input_path).stem}.json"

        # 查找最新的实验结果文件
        spark_results = list(output_path.glob(f"spark_{operator}_*.json"))
        ray_results = list(output_path.glob(f"ray_{operator}_*.json"))

        if not spark_results or not ray_results:
            logger.warning("未找到完整的实验结果文件")
            return

        # 读取最新的结果
        spark_result_path = max(spark_results, key=lambda p: p.stat().st_mtime)
        ray_result_path = max(ray_results, key=lambda p: p.stat().st_mtime)

        from .metrics import load_experiment_result
        spark_result = load_experiment_result(spark_result_path)
        ray_result = load_experiment_result(ray_result_path)

        # 生成对比报告
        comparison = {
            "comparison_id": f"comparison_{operator}_{Path(input_path).stem}",
            "operator": operator,
            "dataset": input_path,
            "timestamp": spark_result.metrics[0].timestamp if spark_result.metrics else None,
            "engines": {
                "spark": {
                    "avg_wall_time": spark_result.avg_wall_time,
                    "std_wall_time": spark_result.std_wall_time,
                    "avg_throughput": spark_result.avg_throughput,
                    "std_throughput": spark_result.std_throughput
                },
                "ray": {
                    "avg_wall_time": ray_result.avg_wall_time,
                    "std_wall_time": ray_result.std_wall_time,
                    "avg_throughput": ray_result.avg_throughput,
                    "std_throughput": ray_result.std_throughput
                }
            },
            "performance_ratio": {
                "wall_time_spark_vs_ray": round(spark_result.avg_wall_time / ray_result.avg_wall_time, 2) if ray_result.avg_wall_time > 0 else None,
                "throughput_spark_vs_ray": round(spark_result.avg_throughput / ray_result.avg_throughput, 2) if ray_result.avg_throughput > 0 else None
            }
        }

        # 保存对比报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"对比报告已保存: {report_path}")

        # 打印对比摘要
        logger.info("对比结果摘要:")
        logger.info(f"Spark平均耗时: {comparison['engines']['spark']['avg_wall_time']:.3f}s")
        logger.info(f"Ray平均耗时: {comparison['engines']['ray']['avg_wall_time']:.3f}s")
        logger.info(f"Spark平均吞吐量: {comparison['engines']['spark']['avg_throughput']:.2f} rows/s")
        logger.info(f"Ray平均吞吐量: {comparison['engines']['ray']['avg_throughput']:.2f} rows/s")
        if comparison["performance_ratio"]["wall_time_spark_vs_ray"]:
            ratio = comparison["performance_ratio"]["wall_time_spark_vs_ray"]
            if ratio > 1:
                logger.info(f"Spark比Ray慢 {ratio:.2f} 倍")
            else:
                logger.info(f"Spark比Ray快 {1/ratio:.2f} 倍")
    except Exception as e:
        logger.error(f"生成对比报告失败: {e}")


def run_pipeline_experiment(args) -> None:
    """运行高性能管道实验"""
    logger = get_logger(__name__)
    try:
        from .pipeline_executor import (
            PipelineConfig,
            OptimizedPipelineRunner
        )

        logger.info(f"运行高性能管道: {' -> '.join(args.operators)}")

        # 确定数据路径和 IO 模式
        data_path = getattr(args, 'data_path', None) or args.input
        
        # 自动选择 IO 模式（如果用户未指定）
        io_mode = _auto_select_io_mode(args)
        is_distributed = _is_distributed_mode(args)
        
        # 如果自动选择了 engine 模式，记录日志
        if not hasattr(args, 'io_mode') or not args.io_mode:
            logger.info(f"自动选择 IO 模式: {io_mode} (分布式模式: {is_distributed})")
        
        # 检查多机模式警告（如果用户明确指定了 pandas 模式）
        if is_distributed and io_mode == 'pandas':
            logger.warning(
                "⚠️  多机模式检测到 io-mode=pandas。"
                "pandas->createDataFrame 会让 driver 先读全量，且输入文件在 worker 不可见。"
                "建议使用 --io-mode engine 并使用共享存储路径（NFS/HDFS/S3）。"
            )

        # 创建管道配置
        pipeline_config = PipelineConfig.from_operator_names(
            operator_names=args.operators,
            engine=args.engine
        )

        # 初始化引擎（计时外）
        spark_session = None
        if args.engine == 'spark':
            from ..engines.spark.session import get_spark
            
            # 解析 Spark 配置
            spark_config = {}
            if hasattr(args, 'spark_conf') and args.spark_conf:
                spark_config = parse_spark_conf(args.spark_conf)
            
            spark_session = get_spark(
                app_name="PipelineApp",
                master=getattr(args, 'spark_master', None),
                config=spark_config if spark_config else None,
                driver_host=getattr(args, 'spark_driver_host', None)
            )
            
            # 根据 io-mode 选择数据加载方式
            if io_mode == 'engine':
                df = load_input_for_engine('spark', data_path, spark=spark_session, is_distributed=is_distributed)
                logger.info("Spark会话已初始化，使用引擎原生读取")
            else:
                pandas_df = load_input_pandas(data_path)
                df = spark_session.createDataFrame(pandas_df)
                logger.info("Spark会话已初始化，数据已转换为Spark DataFrame")
                
        elif args.engine == 'ray':
            from ..engines.ray.runtime import init_ray
            init_ray(
                address=getattr(args, 'ray_address', None),
                namespace=getattr(args, 'ray_namespace', 'benchmark'),
                runtime_env_json=getattr(args, 'ray_runtime_env_json', None)
            )
            
            # 根据 io-mode 选择数据加载方式
            if io_mode == 'engine':
                import ray.data as rd
                df = load_input_for_engine('ray', data_path, spark=None, is_distributed=is_distributed)
                logger.info("Ray运行时已初始化，使用引擎原生读取")
            else:
                pandas_df = load_input_pandas(data_path)
                import ray.data as rd
                df = rd.from_pandas(pandas_df)
                logger.info("Ray运行时已初始化，数据已转换为Ray Dataset")

        # 运行管道实验
        runner = OptimizedPipelineRunner(
            engine=args.engine,
            repeats=args.repeats,
            warmup=args.warmup
        )

        result = runner.run_pipeline_experiment(
            steps=pipeline_config.steps,
            input_df=df,
            spark_session=spark_session
        )

        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存实验结果
        import json
        result_file = output_path / f"pipeline_{args.engine}_{'_'.join(args.operators)}_{int(time.time())}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"管道执行完成，结果已保存到: {result_file}")

        # 打印摘要
        logger.info("性能摘要:")
        logger.info(f"引擎: {args.engine}")
        logger.info(f"管道: {' -> '.join(args.operators)}")
        logger.info(f"平均耗时: {result['avg_time']:.3f}s")
        logger.info(f"标准差: {result['std_time']:.3f}s")
        if result.get('throughput_rows_per_sec'):
            logger.info(f"平均吞吐量: {result['throughput_rows_per_sec']:.2f} rows/s")
        logger.info(f"重复次数: {args.repeats}")

    except Exception as e:
        logger.error(f"管道实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_available_operators(args) -> None:
    """列出可用算子和引擎"""
    logger = get_logger(__name__)
    logger.info("可用算子:")
    operators = list_operator_names()
    if operators:
        for op in operators:
            spec = get_operator_spec(op)
            logger.info(f"  - {op}: {spec.description}")
    else:
        logger.info("  (暂无注册算子)")

    logger.info("可用引擎:")
    logger.info("  - spark: Spark MLlib")
    logger.info("  - ray: Ray Data/Train")


def main() -> None:
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 根据命令行参数设置日志等级
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    py4j_log_level = getattr(logging, args.py4j_log_level.upper(), logging.WARNING)
    setup_logging(level=log_level, force=True, py4j_level=py4j_log_level)

    if not args.command:
        parser.print_help()
        return

    if args.command == 'run':
        run_single_experiment(args)
    elif args.command == 'compare':
        run_comparison_experiment(args)
    elif args.command == 'pipeline':
        run_pipeline_experiment(args)
    elif args.command == 'list':
        list_available_operators(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
