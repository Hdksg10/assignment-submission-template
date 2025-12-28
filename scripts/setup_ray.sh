#!/bin/bash

# Ray环境设置脚本

echo "=== Ray环境设置 ==="

# 检查Python
echo "检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✓ Python版本: $PYTHON_VERSION"
else
    echo "✗ Python3未安装"
    exit 1
fi

# 安装Ray
echo "安装Ray..."
if pip3 install ray==2.0.0 pandas scikit-learn; then
    echo "✓ Ray安装成功"
else
    echo "✗ Ray安装失败"
    exit 1
fi

# 设置环境变量
echo "设置Ray环境变量..."
if [ -z "$RAY_NUM_CPUS" ]; then
    # 检测CPU核心数
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    export RAY_NUM_CPUS="$CPU_CORES"
    echo "✓ RAY_NUM_CPUS: $RAY_NUM_CPUS"
fi

if [ -z "$RAY_NUM_GPUS" ]; then
    export RAY_NUM_GPUS="0"
    echo "✓ RAY_NUM_GPUS: $RAY_NUM_GPUS (如有GPU可修改此值)"
fi

# 测试Ray安装
echo "测试Ray安装..."
if python3 -c "
import ray
ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)
print('Ray版本:', ray.__version__)
print('节点数:', len(ray.nodes()))
resources = ray.cluster_resources()
print('CPU资源:', resources.get('CPU', 0))
ray.shutdown()
print('✓ Ray测试成功')
"; then
    echo "✓ Ray环境设置完成"
else
    echo "✗ Ray测试失败"
    exit 1
fi

echo
echo "Ray环境设置完成！你可以开始使用Ray引擎进行测试。"
