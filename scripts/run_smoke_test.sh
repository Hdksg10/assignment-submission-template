#!/bin/bash

# 冒烟测试脚本
# 确保所有组件能正常工作

set -e  # 遇到错误立即退出

echo "=== Spark MLlib vs Ray 预处理算子对比 - 冒烟测试 ==="
echo

# 检查Python环境
echo "检查Python环境..."
python3 --version
which python3

# 检查必要文件是否存在
echo "检查项目结构..."
REQUIRED_FILES=(
    "data/raw/sample.csv"
    "src/bench/cli.py"
    "src/engines/spark/session.py"
    "src/engines/ray/runtime.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 必需文件 $file 不存在"
        exit 1
    fi
done
echo "✓ 项目文件完整性检查通过"
echo

# 检查数据文件
echo "检查数据文件..."
if [ ! -s "data/raw/sample.csv" ]; then
    echo "错误: sample.csv 文件为空"
    exit 1
fi

# 统计行数
LINES=$(wc -l < data/raw/sample.csv)
if [ "$LINES" -lt 100 ]; then
    echo "警告: sample.csv 行数过少 ($LINES 行)"
else
    echo "✓ 数据文件检查通过 ($LINES 行)"
fi
echo

# 创建输出目录
mkdir -p experiments/reports
mkdir -p experiments/runs

echo "=== 测试Spark引擎 ==="
echo "运行Spark StandardScaler测试..."

# 测试Spark引擎
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from engines.spark.session import get_spark
    spark = get_spark('SmokeTest')
    print('✓ Spark会话创建成功')
    spark.stop()
except ImportError as e:
    print('✗ Spark依赖缺失:', e)
    print('请安装: pip install pyspark')
    sys.exit(1)
except Exception as e:
    print('✗ Spark测试失败:', e)
    sys.exit(1)
"; then
    echo "✓ Spark引擎测试通过"
else
    echo "✗ Spark引擎测试失败"
    exit 1
fi

echo
echo "=== 测试Ray引擎 ==="
echo "运行Ray StandardScaler测试..."

# 测试Ray引擎
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from engines.ray.runtime import init_ray, shutdown_ray
    init_ray()
    print('✓ Ray运行时初始化成功')
    shutdown_ray()
except ImportError as e:
    print('✗ Ray依赖缺失:', e)
    print('请安装: pip install ray')
    sys.exit(1)
except Exception as e:
    print('✗ Ray测试失败:', e)
    sys.exit(1)
"; then
    echo "✓ Ray引擎测试通过"
else
    echo "✗ Ray引擎测试失败"
    exit 1
fi

echo
echo "=== 测试基准组件 ==="
echo "测试算子规格和CLI..."

# 测试基础组件
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from bench.operator_spec import get_operator_spec, list_operator_names
    from bench.io import load_csv
    
    # 测试算子规格
    operators = list_operator_names()
    print(f'✓ 可用算子: {operators}')
    
    if 'StandardScaler' in operators:
        spec = get_operator_spec('StandardScaler')
        print(f'✓ StandardScaler规格加载成功: {spec.description}')
    else:
        print('✗ StandardScaler算子未找到')
        sys.exit(1)
    
    # 测试数据加载
    df = load_csv('data/raw/sample.csv')
    print(f'✓ 数据加载成功: 形状 {df.shape}')
    
except Exception as e:
    print('✗ 基准组件测试失败:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
    echo "✓ 基准组件测试通过"
else
    echo "✗ 基准组件测试失败"
    exit 1
fi

echo
echo "=== 冒烟测试完成 ==="
echo "所有核心组件测试通过！"
echo
echo "接下来你可以运行:"
echo "  python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv"
echo "  python -m src.bench.cli run --engine ray --operator StandardScaler --input data/raw/sample.csv"
echo "  python -m src.bench.cli compare --operator StandardScaler --input data/raw/sample.csv"
