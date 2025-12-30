#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试四个Ray算子在框架中的运行情况
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bench.operator_spec import get_operator_spec
from bench.io import load_csv
from engines.ray.operators import (
    run_standardscaler,
    run_minmaxscaler,
    run_stringindexer,
    run_onehotencoder
)

def test_operators():
    """测试四个算子"""
    print("=" * 80)
    print("开始测试四个Ray算子在框架中的运行")
    print("=" * 80)
    
    # 加载样本数据
    print("\n[1/5] 加载样本数据...")
    data_path = "data/raw/sample.csv"
    df = load_csv(data_path)
    print(f"✓ 数据加载成功: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    print(f"  数据类型: {dict(df.dtypes)}")
    
    # 测试StandardScaler
    print("\n[2/5] 测试StandardScaler...")
    try:
        spec = get_operator_spec("StandardScaler")
        result = run_standardscaler(df, spec)
        print(f"✓ StandardScaler执行成功")
        print(f"  输出列: {[c for c in result.columns if 'scaled' in c]}")
        print(f"  x1_scaled均值: {result['x1_scaled'].mean():.6f}")
        print(f"  x1_scaled标准差: {result['x1_scaled'].std(ddof=0):.6f}")
    except Exception as e:
        print(f"✗ StandardScaler执行失败: {e}")
        return False
    
    # 测试MinMaxScaler
    print("\n[3/5] 测试MinMaxScaler...")
    try:
        spec = get_operator_spec("MinMaxScaler")
        result = run_minmaxscaler(df, spec)
        print(f"✓ MinMaxScaler执行成功")
        print(f"  输出列: {spec.output_cols}")
        print(f"  所有列: {list(result.columns)}")
        print(f"  x1_scaled最小值: {result['x1_scaled'].min():.6f}")
        print(f"  x1_scaled最大值: {result['x1_scaled'].max():.6f}")
    except Exception as e:
        print(f"✗ MinMaxScaler执行失败: {e}")
        return False
    
    # 测试StringIndexer
    print("\n[4/5] 测试StringIndexer...")
    try:
        spec = get_operator_spec("StringIndexer")
        result = run_stringindexer(df, spec)
        print(f"✓ StringIndexer执行成功")
        print(f"  输出列: {[c for c in result.columns if 'indexed' in c]}")
        print(f"  cat_indexed唯一值: {result['cat_indexed'].nunique()}")
        print(f"  cat_indexed值: {sorted(result['cat_indexed'].unique())}")
    except Exception as e:
        print(f"✗ StringIndexer执行失败: {e}")
        return False
    
    # 测试OneHotEncoder（使用StringIndexer的输出）
    print("\n[5/5] 测试OneHotEncoder...")
    try:
        # 首先运行StringIndexer获得indexed列
        spec_si = get_operator_spec("StringIndexer")
        df_indexed = run_stringindexer(df, spec_si)
        
        # 然后运行OneHotEncoder
        spec_ohe = get_operator_spec("OneHotEncoder")
        result = run_onehotencoder(df_indexed, spec_ohe)
        print(f"✓ OneHotEncoder执行成功")
        
        # 检查输出列
        ohe_cols = [c for c in result.columns if 'onehot' in c or c.startswith('cat_indexed_')]
        print(f"  独热编码列数: {len(ohe_cols)}")
        print(f"  独热编码列示例: {ohe_cols[:5] if len(ohe_cols) > 5 else ohe_cols}")
    except Exception as e:
        print(f"✗ OneHotEncoder执行失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ 所有四个算子均在框架中成功运行！")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_operators()
    sys.exit(0 if success else 1)
