#!/usr/bin/env python3
"""
验证基准测试计时边界的脚本

检查所有关键组件是否正确实现了计时边界要求。
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_materialize_module():
    """检查 materialize 模块"""
    print("✓ 检查 materialize 模块...")
    try:
        from bench.materialize import materialize_spark, materialize_ray
        print("  ✓ materialize_spark 存在")
        print("  ✓ materialize_ray 存在")
        return True
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def check_metrics_signature():
    """检查 MetricsCollector 签名"""
    print("✓ 检查 MetricsCollector.collect_metrics 签名...")
    try:
        from bench.metrics import MetricsCollector
        import inspect
        
        sig = inspect.signature(MetricsCollector.collect_metrics)
        params = list(sig.parameters.keys())
        
        required_params = ['input_rows', 'input_cols', 'output_rows', 'output_cols', 'elapsed_seconds']
        for param in required_params:
            if param in params:
                print(f"  ✓ 参数 '{param}' 存在")
            else:
                print(f"  ✗ 参数 '{param}' 缺失")
                return False
        
        # 确保不再接受 DataFrame
        if 'input_df' in params or 'output_df' in params:
            print("  ✗ 仍然接受 DataFrame 参数（应该已移除）")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def check_experiment_runner_signature():
    """检查 ExperimentRunner.run_experiment 签名"""
    print("✓ 检查 ExperimentRunner.run_experiment 签名...")
    try:
        from bench.metrics import ExperimentRunner
        import inspect
        
        sig = inspect.signature(ExperimentRunner.run_experiment)
        params = list(sig.parameters.keys())
        
        if 'input_profile_df' in params:
            print("  ✓ 参数 'input_profile_df' 存在")
        else:
            print("  ✗ 参数 'input_profile_df' 缺失")
            return False
        
        if 'materialize_func' in params:
            print("  ✓ 参数 'materialize_func' 存在")
        else:
            print("  ✗ 参数 'materialize_func' 缺失")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def check_ray_operators():
    """检查 Ray 算子结构"""
    print("✓ 检查 Ray 算子...")
    try:
        # 先检查源代码文件是否存在并包含正确的函数定义
        import ast
        
        operators_to_check = [
            ('standardscaler.py', 'run_standardscaler', 'run_standardscaler_with_ray_data'),
            ('minmaxscaler.py', 'run_minmaxscaler', 'run_minmaxscaler_with_ray_data'),
            ('stringindexer.py', 'run_stringindexer', 'run_stringindexer_with_ray_data'),
            ('onehotencoder.py', 'run_onehotencoder', 'run_onehotencoder_with_ray_data'),
        ]
        
        operators_dir = Path(__file__).parent / "src" / "engines" / "ray" / "operators"
        
        for filename, wrapper_name, core_name in operators_to_check:
            filepath = operators_dir / filename
            if not filepath.exists():
                print(f"  ✗ 文件 {filename} 不存在")
                return False
            
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())
            
            functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            
            if wrapper_name not in functions:
                print(f"  ✗ {filename}: 缺少 {wrapper_name}")
                return False
            
            if core_name not in functions:
                print(f"  ✗ {filename}: 缺少 {core_name}")
                return False
            
            print(f"  ✓ {filename}: wrapper 和 core 函数都存在")
        
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline_executor():
    """检查 Pipeline 执行器"""
    print("✓ 检查 Pipeline 执行器...")
    try:
        from bench.pipeline_executor import HighPerformancePipelineExecutor
        import inspect
        
        sig = inspect.signature(HighPerformancePipelineExecutor.execute_pipeline)
        params = list(sig.parameters.keys())
        
        if 'per_step_timing' in params:
            print("  ✓ 参数 'per_step_timing' 存在")
        else:
            print("  ✗ 参数 'per_step_timing' 缺失")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def main():
    """运行所有检查"""
    print("=" * 60)
    print("基准测试计时边界验证")
    print("=" * 60)
    print()
    
    checks = [
        check_materialize_module,
        check_metrics_signature,
        check_experiment_runner_signature,
        check_ray_operators,
        check_pipeline_executor,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"✗ 检查异常: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("验证结果")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if all(results):
        print("✓ 所有检查通过！")
        return 0
    else:
        print("✗ 部分检查失败，请查看上面的详细信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())

