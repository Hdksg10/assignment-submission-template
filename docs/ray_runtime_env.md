# Ray Runtime Environment 配置指南

## 概述

Ray Runtime Environment 允许在多机集群中自动分发代码依赖、环境变量和文件。这对于确保所有节点使用相同的依赖版本和配置非常重要。

## 何时需要使用 Runtime Environment

在以下情况下，建议使用 Runtime Environment：

1. **多机集群部署**：当连接到远程 Ray 集群时
2. **依赖不一致**：当不同节点的 Python 环境或依赖版本不同时
3. **代码分发**：当需要确保所有节点使用相同的代码版本时
4. **环境变量**：当需要在所有节点设置相同的环境变量时

## 配置方式

### 1. 通过命令行参数

使用 `--ray-runtime-env-json` 参数传递 JSON 格式的配置：

```bash
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --ray-runtime-env-json '{"pip": ["pandas==1.5.3", "scikit-learn==1.2.0"]}'
```

### 2. 通过 Python 代码

```python
from src.engines.ray.runtime import init_ray
import json

runtime_env = {
    "pip": ["pandas==1.5.3", "scikit-learn==1.2.0"],
    "env_vars": {"MY_VAR": "value"}
}

init_ray(
    address="192.168.1.100:6379",
    runtime_env_json=json.dumps(runtime_env)
)
```

## 配置选项

### 1. pip 依赖

指定需要安装的 Python 包：

```json
{
  "pip": ["pandas==1.5.3", "scikit-learn==1.2.0", "numpy>=1.20.0"]
}
```

**注意事项**：
- 包会在每个节点上安装，可能需要一些时间
- 建议固定版本号以确保一致性
- 大型包（如 TensorFlow）安装时间较长

### 2. conda 环境

使用 conda 环境（需要预先创建）：

```json
{
  "conda": "my_env"
}
```

或使用 conda 环境文件：

```json
{
  "conda": {
    "conda_env_name": "my_env",
    "conda_env_file": "environment.yml"
  }
}
```

### 3. 环境变量

设置环境变量：

```json
{
  "env_vars": {
    "PYTHONPATH": "/path/to/code",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "4"
  }
}
```

### 4. 工作目录

指定工作目录（代码路径）：

```json
{
  "working_dir": "/path/to/your/code"
}
```

### 5. 排除文件

排除不需要分发的文件：

```json
{
  "excludes": ["*.pyc", "__pycache__", "*.log"]
}
```

### 6. 完整示例

```json
{
  "pip": [
    "pandas==1.5.3",
    "scikit-learn==1.2.0",
    "numpy>=1.20.0"
  ],
  "env_vars": {
    "PYTHONPATH": "/home/user/project/src",
    "OMP_NUM_THREADS": "4"
  },
  "working_dir": "/home/user/project",
  "excludes": ["*.pyc", "__pycache__", "*.log", "data/"]
}
```

## 常见场景

### 场景 1：确保依赖版本一致

```json
{
  "pip": [
    "pandas==1.5.3",
    "scikit-learn==1.2.0",
    "ray[data]==2.52.1"
  ]
}
```

### 场景 2：分发本地代码

```json
{
  "working_dir": "/home/user/ml-benchmark",
  "excludes": ["data/", "*.csv", "experiments/"]
}
```

### 场景 3：设置 GPU 环境

```json
{
  "pip": ["torch==2.0.0", "torchvision==0.15.0"],
  "env_vars": {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "NCCL_DEBUG": "INFO"
  }
}
```

### 场景 4：混合配置

```json
{
  "pip": ["pandas==1.5.3", "scikit-learn==1.2.0"],
  "conda": "ml-env",
  "env_vars": {
    "PYTHONPATH": "/shared/code",
    "DATA_PATH": "/shared/data"
  },
  "working_dir": "/shared/code",
  "excludes": ["*.pyc", "__pycache__", "*.log"]
}
```

## 最佳实践

1. **固定版本号**：使用精确的版本号（如 `pandas==1.5.3`）而不是范围（如 `pandas>=1.5.0`）

2. **最小化依赖**：只包含必需的包，减少安装时间

3. **使用共享存储**：对于大型代码库，使用共享存储（NFS、S3等）而不是通过 runtime_env 分发

4. **测试配置**：在小型集群上先测试 runtime_env 配置，确保所有依赖正确安装

5. **缓存环境**：Ray 会缓存 runtime_env，相同配置的后续任务会更快启动

6. **错误处理**：如果依赖安装失败，Ray 会抛出异常，检查日志以诊断问题

## 故障排查

### 问题 1：依赖安装失败

**症状**：任务启动失败，错误信息包含 pip/conda 错误

**解决方案**：
- 检查包名和版本号是否正确
- 确认网络连接正常（需要访问 PyPI/conda 仓库）
- 检查节点是否有足够的磁盘空间

### 问题 2：环境变量未生效

**症状**：代码中读取的环境变量为 None 或默认值

**解决方案**：
- 确认 JSON 格式正确（使用 `json.dumps()` 验证）
- 检查环境变量名称拼写
- 确认在任务启动前设置环境变量

### 问题 3：代码未更新

**症状**：修改代码后，节点仍使用旧版本

**解决方案**：
- 清除 Ray 的 runtime_env 缓存
- 使用不同的 `working_dir` 或添加版本标识
- 重启 Ray 集群

## 示例命令

### 基本使用

```bash
# 单机模式（不需要 runtime_env）
python -m src.bench.cli pipeline \
    --engine ray \
    --operators StandardScaler \
    --input data/sample.csv

# 多机模式（需要 runtime_env）
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --ray-runtime-env-json '{"pip": ["pandas==1.5.3", "scikit-learn==1.2.0"]}' \
    --operators StandardScaler \
    --input hdfs://namenode:9000/data/sample.csv
```

### 使用配置文件

创建 `ray_runtime_env.json`：

```json
{
  "pip": [
    "pandas==1.5.3",
    "scikit-learn==1.2.0",
    "numpy>=1.20.0"
  ],
  "env_vars": {
    "PYTHONPATH": "/shared/code"
  }
}
```

使用配置文件：

```bash
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --ray-runtime-env-json "$(cat ray_runtime_env.json)" \
    --operators StandardScaler \
    --input hdfs://namenode:9000/data/sample.csv
```

## 参考资源

- [Ray Runtime Environment 官方文档](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments)
- [Ray 多机集群部署指南](https://docs.ray.io/en/latest/cluster/getting-started.html)

