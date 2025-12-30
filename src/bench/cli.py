"""
命令行接口

提供统一的实验运行和对比接口。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json

# 导入项目模块
from .operator_spec import get_operator_spec, list_operator_names
from .io import load_csv, save_csv, get_file_info
from .metrics import ExperimentRunner, save_experiment_result
from .logger import get_logger, setup_logging
import logging


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
    run_parser.add_argument('--output', required=True,
                           help='输出目录路径')
    run_parser.add_argument('--repeats', type=int, default=3,
                           help='重复运行次数 (默认: 3)')
    run_parser.add_argument('--warmup', action='store_true', default=True,
                           help='是否执行预热运行 (默认: True)')
    run_parser.add_argument('--params', type=json.loads, default={},
                           help='额外的算子参数 (JSON格式)')

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
    pipeline_parser.add_argument('--output', required=True,
                                help='输出目录路径')
    pipeline_parser.add_argument('--repeats', type=int, default=3,
                                help='重复运行次数 (默认: 3)')
    pipeline_parser.add_argument('--warmup', action='store_true', default=True,
                                help='是否执行预热运行 (默认: True)')
    pipeline_parser.add_argument('--params', type=json.loads, default={},
                                help='算子参数 (JSON格式，key为算子名)')

    return parser


def run_single_experiment(args) -> None:
    """运行单次实验"""
    logger = get_logger(__name__)
    try:
        # 获取算子规格
        spec = get_operator_spec(args.operator)
        logger.info(f"运行算子: {spec.name}")
        logger.info(f"描述: {spec.description}")

        # 加载数据（pandas，用于profile）
        logger.info(f"加载数据: {args.input}")
        df = load_csv(args.input)
        logger.info(f"数据形状: {df.shape}")

        # 合并参数
        operator_params = spec.params.copy()
        operator_params.update(args.params)

        # 动态导入引擎模块
        if args.engine == 'spark':
            from ..engines.spark.session import get_spark
            from ..engines.spark.operators import run_standardscaler
            from .materialize import materialize_spark

            # 初始化Spark（计时外）
            spark = get_spark("BenchmarkApp")
            logger.info("Spark会话已初始化")

            # 转换pandas到Spark DataFrame（计时外）
            spark_df = spark.createDataFrame(df)
            logger.info("数据已转换为Spark DataFrame")

            # 运行算子
            if args.operator == 'StandardScaler':
                runner = ExperimentRunner(repeats=args.repeats, warmup=args.warmup)
                result = runner.run_experiment(
                    engine=args.engine,
                    operator=args.operator,
                    dataset_path=args.input,
                    operator_func=run_standardscaler,
                    input_profile_df=df,  # pandas用于profile
                    materialize_func=materialize_spark,  # 触发执行
                    spark=spark,
                    input_df=spark_df,  # Spark DF作为算子输入
                    spec=spec
                )
            else:
                raise ValueError(f"不支持的算子: {args.operator}")

        elif args.engine == 'ray':
            import ray
            import ray.data as rd
            from ..engines.ray.runtime import init_ray
            from ..engines.ray.operators import run_standardscaler
            from .materialize import materialize_ray

            # 初始化Ray（计时外）
            init_ray()
            logger.info("Ray运行时已初始化")

            # 转换pandas到Ray Dataset（计时外）
            ray_ds = rd.from_pandas(df)
            logger.info("数据已转换为Ray Dataset")

            # 运行算子
            if args.operator == 'StandardScaler':
                runner = ExperimentRunner(repeats=args.repeats, warmup=args.warmup)
                result = runner.run_experiment(
                    engine=args.engine,
                    operator=args.operator,
                    dataset_path=args.input,
                    operator_func=run_standardscaler,
                    input_profile_df=df,  # pandas用于profile
                    materialize_func=materialize_ray,  # 触发执行
                    input_df=ray_ds,  # Ray Dataset作为算子输入
                    spec=spec
                )
            else:
                raise ValueError(f"不支持的算子: {args.operator}")

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

        # 创建临时参数对象用于Spark
        spark_args = argparse.Namespace()
        spark_args.engine = 'spark'
        spark_args.operator = args.operator
        spark_args.input = args.input
        spark_args.output = args.output
        spark_args.repeats = args.repeats
        spark_args.warmup = True
        spark_args.params = {}

        # 创建临时参数对象用于Ray
        ray_args = argparse.Namespace()
        ray_args.engine = 'ray'
        ray_args.operator = args.operator
        ray_args.input = args.input
        ray_args.output = args.output
        ray_args.repeats = args.repeats
        ray_args.warmup = True
        ray_args.params = {}

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

        # 加载数据
        logger.info(f"加载数据: {args.input}")
        df = load_csv(args.input)
        logger.info(f"数据形状: {df.shape}")

        # 创建管道配置
        pipeline_config = PipelineConfig.from_operator_names(
            operator_names=args.operators,
            engine=args.engine
        )

        # 保存原始pandas DataFrame用于profile
        pandas_df = df

        # 初始化引擎（计时外）
        spark_session = None
        if args.engine == 'spark':
            from ..engines.spark.session import get_spark
            spark_session = get_spark("PipelineApp")
            df = spark_session.createDataFrame(df)
            logger.info("Spark会话已初始化，数据已转换为Spark DataFrame")
        elif args.engine == 'ray':
            from ..engines.ray.runtime import init_ray
            init_ray()
            # 对于Ray，使用Ray Dataset（计时外）
            import ray.data as rd
            df = rd.from_pandas(df)
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
