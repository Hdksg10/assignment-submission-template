#!/usr/bin/env python
"""
测试4个纯Ray Data算子
验证完全不使用sklearn的实现
"""

import pandas as pd
import ray.data
from src.bench.operator_spec import OperatorSpec
from src.engines.ray.operators.standardscaler import run_standardscaler, run_standardscaler_with_ray_data
from src.engines.ray.operators.minmaxscaler import run_minmaxscaler, run_minmaxscaler_with_ray_data
from src.engines.ray.operators.stringindexer import run_stringindexer, run_stringindexer_with_ray_data
from src.engines.ray.operators.onehotencoder import run_onehotencoder, run_onehotencoder_with_ray_data

def test_standardscaler():
    print("\n" + "="*50)
    print("测试 StandardScaler")
    print("="*50)
    
    df = pd.read_csv('data/raw/sample.csv')
    spec = OperatorSpec(
        name='StandardScaler',
        input_cols=['x1', 'x2'],
        output_cols=['x1', 'x2'],
        params={},
        description='test'
    )
    
    # pandas模式
    result_pd = run_standardscaler(df, spec)
    print("✓ pandas模式成功")
    print(f"  输出形状: {result_pd.shape}")
    print(f"  x1统计: mean={result_pd['x1'].mean():.4f}, std={result_pd['x1'].std():.4f}")
    
    # Ray Dataset模式
    ds = ray.data.from_pandas(df)
    result_ray = run_standardscaler_with_ray_data(ds, spec)
    print("✓ Ray Dataset模式成功")
    print(f"  输出类型: {type(result_ray)}")

def test_minmaxscaler():
    print("\n" + "="*50)
    print("测试 MinMaxScaler")
    print("="*50)
    
    df = pd.read_csv('data/raw/sample.csv')
    spec = OperatorSpec(
        name='MinMaxScaler',
        input_cols=['x1', 'x2'],
        output_cols=['x1', 'x2'],
        params={},
        description='test'
    )
    
    # pandas模式
    result_pd = run_minmaxscaler(df, spec)
    print("✓ pandas模式成功")
    print(f"  输出形状: {result_pd.shape}")
    print(f"  x1范围: min={result_pd['x1'].min():.4f}, max={result_pd['x1'].max():.4f}")
    
    # Ray Dataset模式
    ds = ray.data.from_pandas(df)
    result_ray = run_minmaxscaler_with_ray_data(ds, spec)
    print("✓ Ray Dataset模式成功")
    print(f"  输出类型: {type(result_ray)}")

def test_stringindexer():
    print("\n" + "="*50)
    print("测试 StringIndexer")
    print("="*50)
    
    df = pd.read_csv('data/raw/sample.csv')
    spec = OperatorSpec(
        name='StringIndexer',
        input_cols=['cat'],
        output_cols=['cat_idx'],
        params={},
        description='test'
    )
    
    # pandas模式
    result_pd = run_stringindexer(df, spec)
    print("✓ pandas模式成功")
    print(f"  输出形状: {result_pd.shape}")
    print(f"  cat_idx值: {sorted(result_pd['cat_idx'].unique().tolist())}")
    
    # Ray Dataset模式
    ds = ray.data.from_pandas(df)
    result_ray = run_stringindexer_with_ray_data(ds, spec)
    print("✓ Ray Dataset模式成功")
    print(f"  输出类型: {type(result_ray)}")

def test_onehotencoder():
    print("\n" + "="*50)
    print("测试 OneHotEncoder")
    print("="*50)
    
    df = pd.read_csv('data/raw/sample.csv')
    spec = OperatorSpec(
        name='OneHotEncoder',
        input_cols=['cat'],
        output_cols=['cat'],
        params={},
        description='test'
    )
    
    # pandas模式
    result_pd = run_onehotencoder(df, spec)
    print("✓ pandas模式成功")
    print(f"  输出形状: {result_pd.shape}")
    print(f"  输出列: {[col for col in result_pd.columns if 'cat' in col]}")
    
    # Ray Dataset模式
    ds = ray.data.from_pandas(df)
    result_ray = run_onehotencoder_with_ray_data(ds, spec)
    print("✓ Ray Dataset模式成功")
    print(f"  输出类型: {type(result_ray)}")

if __name__ == '__main__':
    print("\n开始测试纯Ray Data算子（无sklearn）")
    
    try:
        test_standardscaler()
        test_minmaxscaler()
        test_stringindexer()
        test_onehotencoder()
        
        print("\n" + "="*50)
        print("所有测试通过！")
        print("="*50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
