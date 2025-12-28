# Spark MLlib vs Ray 预处理算子对比项目

## 项目概述

本项目是一个专业的基准测试框架，用于对比 Spark MLlib 与 Ray（Ray Data / Ray Train 生态）在机器学习预处理算子上的功能一致性与性能表现。项目提供了统一的接口、高性能的执行引擎和完整的实验工具链，支持单算子测试、多算子管道测试以及跨引擎性能对比。

## 项目目的

- **功能一致性验证**：确保 Spark MLlib 和 Ray 在相同预处理算子上的输出结果一致
- **性能基准测试**：对比两个框架在相同算子上的执行性能，包括吞吐量、延迟等指标
- **可扩展性评估**：测试不同数据规模下的性能表现
- **开发工具支持**：为研究人员和开发者提供便捷的实验工具和开发框架

## 仓库结构

```
.
├── README.md              # 项目说明文档
├── pyproject.toml         # Python包配置
├── .gitignore            # Git忽略规则
├── .editorconfig         # 编辑器配置
├── .gitattributes        # Git属性配置
├── .env.example          # 环境变量示例
├── docs/                 # 项目文档
│   ├── design.md         # 设计原则
│   ├── operators.md      # 算子规格说明
│   ├── datasets.md       # 数据集说明
│   └── experiment_protocol.md  # 实验协议
├── src/                  # 源代码
│   ├── bench/            # 基准测试核心
│   │   ├── operator_spec.py      # 算子规格定义
│   │   ├── operator_executor.py  # 高性能算子执行器工厂
│   │   ├── pipeline_executor.py  # 高性能管道执行器
│   │   ├── metrics.py            # 指标采集
│   │   ├── io.py                 # 数据IO
│   │   ├── logger.py             # 日志系统
│   │   └── cli.py                # 命令行接口
│   └── engines/          # 引擎实现
│       ├── spark/        # Spark MLlib实现
│       │   ├── session.py     # Spark会话管理
│       │   └── operators/     # Spark算子实现
│       └── ray/          # Ray实现
│           ├── runtime.py     # Ray运行时管理
│           └── operators/     # Ray算子实现
├── experiments/          # 实验相关
│   ├── configs/          # 实验配置
│   ├── runs/             # 实验运行数据（不提交）
│   └── reports/          # 实验报告（小文件可提交）
├── data/                 # 数据管理
│   ├── raw/              # 原始数据（小样本）
│   ├── processed/        # 处理后数据（不提交）
│   └── README.md         # 数据管理说明
├── requirements.txt      # Python依赖（完整版）
├── requirements-dev.txt  # 开发环境依赖
├── requirements-minimal.txt  # 最小化依赖
├── scripts/              # 工具脚本
│   ├── setup_spark.sh    # Spark环境设置
│   ├── setup_ray.sh      # Ray环境设置
│   ├── download_datasets.sh  # 数据集下载
│   └── run_smoke_test.sh     # 冒烟测试
└── tests/                # 测试代码
    ├── test_operator_contracts.py  # 算子契约测试
    └── test_smoke.py               # 冒烟测试
```

## 快速开始

### 环境要求

**必需环境**：
- Python 3.8 或更高版本
- pip 包管理器

**可选环境**（根据使用的引擎选择）：
- **Spark 引擎**：Java 8+ 和 Spark 3.3+（或通过 PySpark 安装，推荐Spark3.5.7）
- **Ray 引擎**：Ray 2.0+（通过 pip 安装）

**推荐环境**：
- 操作系统：Linux / macOS / Windows (WSL2)
- 内存：至少 4GB 可用内存（用于运行基准测试）
- 存储：至少 1GB 可用空间（用于数据和实验结果）

### 安装依赖

#### 方式1：开发模式安装（推荐！！！）

```bash
# 安装包及其依赖到开发环境
pip install -e .

# 如需开发工具，额外安装
pip install -r requirements-dev.txt
```

#### 方式2：使用requirements文件（不推荐）

```bash
# 完整安装（包含所有功能和引擎）
pip install -r requirements.txt

# 最小化安装（只安装核心依赖）
pip install -r requirements-minimal.txt

# 开发环境安装（包含测试和开发工具）
pip install -r requirements-dev.txt
```

#### 方式3：手动安装（不推荐）

```bash
# 核心依赖
pip install pandas numpy pyarrow scikit-learn

# 选择引擎（根据需要安装一个或两个）
pip install pyspark==3.5.7      # Spark MLlib引擎
pip install ray[default]>=2.0.0  # Ray引擎

# 可选：开发和可视化工具
pip install pytest jupyter matplotlib seaborn
```

### 验证安装

安装完成后，验证项目是否正确设置：

```bash
# 查看可用算子和引擎
python -m src.bench.cli list
```

### 运行冒烟测试

运行冒烟测试确保环境配置正确：

```bash
# 确保在仓库根目录
bash scripts/run_smoke_test.sh
```

### 运行单次基准测试

使用 `run` 命令运行单个算子的基准测试：

```bash
# Spark引擎运行StandardScaler算子
python -m src.bench.cli run \
    --engine spark \
    --operator StandardScaler \
    --input data/raw/sample.csv \
    --output experiments/runs/ \
    --repeats 3 \
    --warmup

# Ray引擎运行StandardScaler算子
python -m src.bench.cli run \
    --engine ray \
    --operator StandardScaler \
    --input data/raw/sample.csv \
    --output experiments/runs/ \
    --repeats 3 \
    --warmup
```

### 运行高性能管道测试（推荐）

使用 `pipeline` 命令运行多算子管道，这是推荐的方式，具有更低的性能开销：

```bash
# 单算子管道（Spark引擎）
python -m src.bench.cli pipeline \
    --engine spark \
    --operators StandardScaler \
    --input data/raw/sample.csv \
    --output experiments/runs/ \
    --repeats 5 \
    --warmup

# 多算子管道（Ray引擎）
python -m src.bench.cli pipeline \
    --engine ray \
    --operators StandardScaler StringIndexer \
    --input data/raw/sample.csv \
    --output experiments/runs/
```

### 对比测试

使用 `compare` 命令自动运行两个引擎并生成对比报告：

```bash
# 自动运行Spark和Ray，并生成对比报告
python -m src.bench.cli compare \
    --operator StandardScaler \
    --input data/raw/sample.csv \
    --output experiments/reports/ \
    --repeats 3
```

对比报告将保存在 `experiments/reports/` 目录下，包含两个引擎的性能指标对比和性能比率分析。

### 日志配置

项目支持灵活的日志级别配置，可以控制输出详细程度，特别是减少 Spark 通信日志的噪音。

#### CLI 日志配置

所有 CLI 命令都支持 `--log-level` 和 `--py4j-log-level` 参数：

```bash
# 默认：主日志INFO，Py4J日志WARNING（推荐，减少Spark通信日志）
python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv --output experiments/runs/

# 使用DEBUG级别查看详细日志（但Py4J仍为WARNING，减少噪音）
python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv --output experiments/runs/ --log-level DEBUG

# 如果需要调试PySpark通信问题，可以启用Py4J DEBUG日志
python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv --output experiments/runs/ --log-level DEBUG --py4j-log-level DEBUG

# 完全静默Py4J日志（只显示ERROR及以上）
python -m src.bench.cli run --engine spark --operator StandardScaler --input data/raw/sample.csv --output experiments/runs/ --py4j-log-level ERROR
```

**日志级别说明**：
- `DEBUG`: 最详细，包含所有调试信息
- `INFO`: 一般信息（默认）
- `WARNING`: 警告信息（Py4J默认，减少Spark通信日志）
- `ERROR`: 只显示错误
- `CRITICAL`: 只显示严重错误

**为什么需要单独设置Py4J日志级别？**

Py4J 是 PySpark 与 JVM 通信的底层库，在 DEBUG 级别下会产生大量通信日志（如 `Answer received: !yro54`、`Command to send: c` 等）。这些日志对日常使用没有帮助，但会显著增加日志文件大小。默认将 Py4J 日志级别设置为 `WARNING` 可以：
- 减少日志文件大小
- 提高日志可读性
- 保留重要的警告和错误信息

#### 测试日志配置

运行测试时，可以通过环境变量设置日志级别：

```bash
# 默认：主日志DEBUG，Py4J日志WARNING
pytest tests/

# 自定义主日志级别
LOG_LEVEL=INFO pytest tests/test_smoke.py

# 同时自定义Py4J日志级别
LOG_LEVEL=DEBUG PY4J_LOG_LEVEL=WARNING pytest tests/
LOG_LEVEL=DEBUG PY4J_LOG_LEVEL=DEBUG pytest tests/  # 查看所有日志包括Py4J通信

# 在Windows PowerShell中
$env:LOG_LEVEL="DEBUG"; $env:PY4J_LOG_LEVEL="WARNING"; pytest tests/
```

**环境变量说明**：
- `LOG_LEVEL`: 主日志级别（默认：测试时为 `DEBUG`，CLI 时为 `INFO`）
- `PY4J_LOG_LEVEL`: Py4J 通信日志级别（默认：`WARNING`）

#### 日志文件位置

日志文件保存在 `logs/` 目录下，按日期命名：
- `logs/benchmark_2025-12-28.log` - 当天的日志文件

## 结果产物位置

项目运行后会在以下目录生成结果：

- **实验报告**: `experiments/reports/` - 包含JSON格式的性能指标和对比结果，建议提交到Git用于版本控制
- **运行数据**: `experiments/runs/` - 临时运行文件，包含详细的执行日志和中间结果，不提交到Git
- **处理数据**: `data/processed/` - 算子输出结果，不提交到Git
- **日志文件**: `logs/` - 按日期轮转的日志文件，格式为 `benchmark_YYYY-MM-DD.log`

## 核心功能特性

### 高性能执行器系统

项目实现了高性能的算子执行器系统，具有以下特点：

- **零开销算子查找**：预注册机制，模块导入时完成注册，运行时O(1)查找
- **最小化包装开销**：直接函数调用，包装开销 < 1%，确保测试结果与直接调用API一致
- **高精度性能测量**：使用 `time.perf_counter()` 提供纳秒级精度，不受系统时钟影响
- **智能执行触发**：自动处理 Spark/Ray 的 lazy execution 特性

详细说明请参考 [README_HIGH_PERFORMANCE_EXECUTOR.md](README_HIGH_PERFORMANCE_EXECUTOR.md)。

### 统一接口设计

- **引擎无关**：相同的代码可以运行在 Spark 和 Ray 上
- **配置驱动**：通过配置定义算子管道，无需修改核心逻辑
- **类型安全**：完整的类型注解，提供良好的IDE支持

## 开发指南

### 开发环境设置

在开始开发前，请确保已安装项目包：

```bash
# 在仓库根目录执行
pip install -e .
pip install -r requirements-dev.txt
```

这将安装包到开发环境，使导入正常工作并提供开发工具。

### 添加新算子

添加新算子需要完成以下步骤：

1. **定义算子规格**：在 `src/bench/operator_spec.py` 中使用 `register_operator_spec()` 注册算子规格
   ```python
   register_operator_spec(OperatorSpec(
       name="MinMaxScaler",
       input_cols=["x1", "x2"],
       output_cols=["x1_scaled", "x2_scaled"],
       params={"min": 0.0, "max": 1.0},
       description="最小最大标准化"
   ))
   ```

2. **实现Spark版本**：在 `src/engines/spark/operators/` 中创建算子实现文件
   - 函数签名：`def run_<operator_name>(spark, input_df: DataFrame, spec: OperatorSpec) -> DataFrame`
   - 在 `src/engines/spark/operators/__init__.py` 中导入并注册

3. **实现Ray版本**：在 `src/engines/ray/operators/` 中创建算子实现文件
   - 函数签名：`def run_<operator_name>_with_ray_data(ray_dataset, spec: OperatorSpec)`
   - 在 `src/engines/ray/operators/__init__.py` 中导入并注册

4. **更新文档**：在 `docs/operators.md` 中添加算子说明和使用示例

5. **添加测试**：在 `tests/` 中添加相应的测试用例，确保功能一致性和正确性

详细实现指南请参考现有算子的实现（如 `StandardScaler`）和 [README_HIGH_PERFORMANCE_EXECUTOR.md](README_HIGH_PERFORMANCE_EXECUTOR.md) 中的扩展指南。

### 实验配置

实验配置位于 `experiments/configs/` 目录，可自定义：
- 数据集路径
- 算子参数
- 重复运行次数
- 性能指标收集

### 日志系统

项目使用统一的日志系统（`src/bench/logger.py`），支持：

1. **灵活的日志级别控制**：可以单独设置主日志和 Py4J 通信日志的级别
2. **自动日志文件管理**：日志自动保存到 `logs/` 目录，按日期轮转
3. **统一的日志格式**：所有日志使用统一的格式，便于分析和调试

**开发时建议**：
- 日常开发使用 `INFO` 级别，Py4J 保持 `WARNING`
- 调试问题时使用 `DEBUG` 级别，但保持 Py4J 为 `WARNING` 以减少噪音
- 只有在调试 PySpark 通信问题时才将 Py4J 设置为 `DEBUG`

## 依赖管理

项目使用现代Python包管理：

- **`pyproject.toml`**: 包配置和依赖声明，支持开发模式安装
- **`requirements.txt`**: 完整依赖，包含所有功能、引擎和常用工具
- **`requirements-dev.txt`**: 开发环境依赖，额外包含测试、代码质量、文档等开发工具
- **`requirements-minimal.txt`**: 最小化依赖，只包含运行基准测试的核心组件

**推荐开发方式**：使用 `pip install -e .` 进行开发模式安装，这将正确设置包结构并支持热重载。

### 传统requirements文件

- **`requirements.txt`**: 完整依赖，包含所有功能、引擎和常用工具
- **`requirements-dev.txt`**: 开发环境依赖，额外包含测试、代码质量、文档等开发工具
- **`requirements-minimal.txt`**: 最小化依赖，只包含运行基准测试的核心组件

### 引擎选择

`requirements-minimal.txt` 允许你选择只安装需要的引擎：

```bash
# 只使用Spark MLlib
pip install pandas numpy pyarrow scikit-learn pyspark==3.5.7

# 只使用Ray
pip install pandas numpy pyarrow scikit-learn ray[default]

# 两者都安装（用于对比测试）
pip install pandas numpy pyarrow scikit-learn pyspark==3.5.7 ray[default]
```

## 注意事项

### 开发环境

- **包安装**：开发时必须先执行 `pip install -e .` 安装包，否则导入会失败
- **导入方式**：使用绝对导入（如 `from bench.operator_spec import ...`），避免相对导入
- **环境隔离**：建议使用虚拟环境（conda/venv）进行开发，避免依赖冲突

### 代码规范

- **算子实现**：所有算子实现应保证输出Schema的一致性，确保Spark和Ray版本输出相同
- **类型注解**：所有公共函数应包含完整的类型注解
- **错误处理**：算子实现应包含适当的错误处理和日志记录

### 性能测试

- **重复次数**：性能测试默认运行3次取平均值，可通过 `--repeats` 参数调整
- **预热运行**：默认启用预热运行（`--warmup`），确保JIT编译和缓存生效
- **日志级别**：生产环境建议使用 `INFO` 级别，调试时使用 `DEBUG` 级别

### 数据管理

- **大数据集**：请使用下载脚本（`scripts/download_datasets.sh`），不要直接提交到仓库
- **实验结果**：实验结果请提交到 `experiments/reports/` 用于版本控制
- **临时文件**：`experiments/runs/` 和 `data/processed/` 目录中的文件不应提交到Git

### 引擎选择

- **Spark引擎**：需要Java环境，适合大规模数据处理和已有Spark集群的场景
- **Ray引擎**：纯Python实现，启动更快，适合快速原型开发和中小规模数据
- **对比测试**：使用 `compare` 命令可以自动运行两个引擎并生成对比报告
