# 分布式运行能力修复总结

本文档总结了针对项目分布式运行能力的修复和改进。

## 修复的问题

### 1. ✅ Ray 初始化时的集群连接检查缺失

**问题描述**：
- 如果 Ray 已经初始化，代码会直接返回，不检查是否连接到正确的集群
- 在多机模式下，如果之前连接到本地集群，现在想连接到远程集群会失败

**修复方案**：
- 添加了 `_normalize_ray_address()` 函数来规范化地址格式以便比较
- 在 `init_ray()` 中检查当前连接的集群地址
- 如果目标地址与当前地址不同，先关闭当前连接再重新初始化
- 添加了 `_validate_ray_cluster_connection()` 函数验证多机模式连接

**文件修改**：
- `src/engines/ray/runtime.py`

### 2. ✅ 文件路径验证缺失

**问题描述**：
- 代码注释要求使用共享存储路径，但没有实际验证
- 在多机模式下，如果使用本地路径，worker 节点无法访问

**修复方案**：
- 添加了 `_is_shared_storage_path()` 函数检测共享存储路径
- 添加了 `_validate_data_path_for_distributed()` 函数验证路径
- 支持检测 HDFS、S3、Azure Blob、NFS 等常见共享存储
- 在多机模式下，对本地路径发出警告或错误

**文件修改**：
- `src/bench/data_ingest.py`
- `load_input_for_engine()` 函数添加了 `is_distributed` 参数

### 3. ✅ 默认 IO 模式不适合多机

**问题描述**：
- 默认 IO 模式是 `pandas`，在多机模式下不合适
- 虽然有警告，但用户可能忽略

**修复方案**：
- 添加了 `_is_distributed_mode()` 函数检测分布式模式
- 添加了 `_auto_select_io_mode()` 函数根据集群模式自动选择 IO 模式
- 多机模式自动使用 `engine` 模式，单机模式使用 `pandas` 模式
- 用户仍可手动指定 `--io-mode` 覆盖自动选择

**文件修改**：
- `src/bench/cli.py`
- `--io-mode` 参数的默认值改为 `None`（自动选择）

### 4. ✅ Spark Driver Host 验证缺失

**问题描述**：
- 虽然有 `driver_host` 参数，但没有验证其有效性
- 在多机模式下，如果 driver host 不可达，会导致连接失败

**修复方案**：
- 添加了 `_validate_driver_host()` 函数验证 driver host
- 使用 `socket.gethostbyname()` 验证主机名/IP 地址格式
- 提供清晰的错误信息帮助用户诊断问题

**文件修改**：
- `src/engines/spark/session.py`

### 5. ✅ Ray Runtime Environment 配置文档不足

**问题描述**：
- 虽然有 `runtime_env_json` 参数，但缺少配置示例和文档
- 在多机模式下，代码依赖需要正确配置

**修复方案**：
- 创建了详细的配置文档 `docs/ray_runtime_env.md`
- 包含配置选项说明、常见场景示例、最佳实践和故障排查

**新增文件**：
- `docs/ray_runtime_env.md`

### 6. ✅ 缺少 Ray 多机模式的连接验证

**问题描述**：
- 虽然代码会打印集群信息，但没有验证是否真正连接到多机集群
- 例如，如果指定了多机地址但只检测到 1 个节点，应该发出警告

**修复方案**：
- 在 `_validate_ray_cluster_connection()` 中验证节点数
- 如果指定了多机模式但只检测到 1 个节点，发出警告
- 检查活跃节点数，如果部分节点离线，发出警告
- 验证集群地址的可达性（对于具体地址，不包括 'auto'）

**文件修改**：
- `src/engines/ray/runtime.py`

## 使用示例

### 自动选择 IO 模式

```bash
# 单机模式 - 自动使用 pandas
python -m src.bench.cli pipeline \
    --engine ray \
    --operators StandardScaler \
    --input data/sample.csv

# 多机模式 - 自动使用 engine
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --operators StandardScaler \
    --input hdfs://namenode:9000/data/sample.csv
```

### 手动指定 IO 模式

```bash
# 即使多机模式，也可以强制使用 pandas（会收到警告）
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --io-mode pandas \
    --operators StandardScaler \
    --input data/sample.csv
```

### 使用 Ray Runtime Environment

```bash
python -m src.bench.cli pipeline \
    --engine ray \
    --ray-address "192.168.1.100:6379" \
    --ray-runtime-env-json '{"pip": ["pandas==1.5.3", "scikit-learn==1.2.0"]}' \
    --operators StandardScaler \
    --input hdfs://namenode:9000/data/sample.csv
```

### 使用 Spark 多机模式

```bash
python -m src.bench.cli pipeline \
    --engine spark \
    --spark-master "spark://master:7077" \
    --spark-driver-host "192.168.1.50" \
    --operators StandardScaler \
    --input hdfs://namenode:9000/data/sample.csv
```

## 验证和测试

### 验证 Ray 集群连接

运行命令后，检查日志输出：

```
Ray运行时初始化成功
版本: 2.52.1
命名空间: benchmark
集群地址: 192.168.1.100:6379
✓ 已连接到 Ray 集群（多机模式）
节点数: 3 (活跃: 3)
集群资源: {'CPU': 12.0, 'GPU': 0.0, ...}
```

如果只检测到 1 个节点，会看到警告：

```
⚠️  警告：指定了多机模式 (address=192.168.1.100:6379)，但只检测到 1 个节点。
这可能表示：
  1. 集群尚未完全启动
  2. 网络连接问题
  3. 实际运行在本地模式
```

### 验证文件路径

在多机模式下使用本地路径，会看到警告：

```
⚠️  警告：检测到本地文件路径 '/local/data/sample.csv' 在多机模式下使用。
在多机集群中，worker节点可能无法访问本地路径。
建议使用共享存储路径，例如：
  - HDFS: hdfs://namenode:port/path/to/file.csv
  - S3: s3://bucket/path/to/file.csv
  - NFS: /shared/data/file.csv (确保所有节点可访问)
  - 或其他分布式文件系统路径
```

### 验证 Spark Driver Host

如果 driver host 不可达，会看到错误：

```
错误：无法解析Driver主机地址 'invalid-host'。
原因: [Errno -2] Name or service not known
请检查：
  1. 主机名或IP地址是否正确
  2. DNS配置是否正确
  3. 网络连接是否正常
```

## 向后兼容性

所有修复都保持了向后兼容性：

1. **IO 模式自动选择**：如果用户明确指定了 `--io-mode`，会使用用户指定的值
2. **文件路径验证**：只发出警告，不会阻止执行（除非路径不存在且不是共享存储）
3. **Ray 初始化**：如果已经连接到正确的集群，会跳过重新初始化
4. **Spark Driver Host**：如果验证失败，只发出警告，不会阻止执行

## 相关文档

- [Ray Runtime Environment 配置指南](ray_runtime_env.md)
- [算子规格说明](operators.md)
- [高性能执行器文档](high_performance_executor.md)

