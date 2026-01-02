# 分布式机器学习预处理算子对比

## 仓库结构

```
.
├── .gitignore             
├── .editorconfig          
├── .gitattributes         
├── .env.example           # 环境变量示例模板，用于配置运行时参数
├── pyproject.toml         # Python包配置文件，定义项目元数据和依赖
├── README.md              # 项目说明文档
├── README_DEVELOP.md      # 开发者指南，包含开发规范、架构设计和扩展说明
├── README_HIGH_PERFORMANCE_EXECUTOR.md  # 高性能执行器详细设计文档
├── requirements.txt       # 完整依赖列表，包含所有功能和引擎依赖
├── requirements-dev.txt    # 开发环境依赖，包含测试、代码质量工具等
├── requirements-minimal.txt # 最小化依赖，仅包含核心组件
├── code/                  
├── data/                  # 数据管理目录
│   ├── README.md         # 数据管理规范文档，说明数据存储和使用规则
│   ├── raw/              # 原始数据存储目录，仅存放小样本测试数据
│   └── processed/        # 处理后数据目录，运行时生成，不提交到Git
├── docs/                  # 项目文档目录
│   ├── design.md         # 系统设计原则和架构说明
│   ├── operators.md      # 各算子的详细规格说明和使用示例
│   ├── datasets.md       # 支持的数据集说明和格式规范
│   ├── experiment_protocol.md  # 实验协议，定义实验标准和流程
│   ├── BENCHMARK_TIMING_CHANGES.md  # 基准测试时序变更记录
│   ├── BENCHMARK_TRIGGER_IMPROVEMENT.md  # 基准测试触发机制改进说明
│   ├── COLUMN_CONSISTENCY_FIX.md  # 列一致性修复文档
│   ├── FINAL_SIMPLIFIED_SOLUTION.md  # 最终简化方案说明
│   ├── SIMPLIFIED_APPROACH.md  # 简化方法说明
│   ├── TEST_FIXES_SUMMARY.md  # 测试修复总结
│   ├── TEST_IMPROVEMENTS.md  # 测试改进说明
│   ├── TIMING_BOUNDARIES_GUIDE.md  # 时序边界指南
│   ├── distributed_fixes.md  # 分布式修复文档
│   ├── high_performance_executor.md  # 高性能执行器详细说明
│   └── ray_runtime_env.md  # Ray运行时环境配置说明
├── scripts/              # 工具脚本目录
│   ├── setup_spark.sh    # Spark环境初始化和配置脚本
│   ├── setup_ray.sh      # Ray环境初始化和配置脚本
│   ├── download_datasets.sh  # 数据集下载脚本，支持多种数据源
│   ├── run_smoke_test.sh     # 冒烟测试脚本，验证基本功能
│   ├── debug_consistency.py  # 一致性调试工具
│   ├── test_consistency_fix.py  # 一致性修复测试脚本
│   └── verify_timing_boundaries.py  # 时序边界验证工具
├── src/                  # 源代码目录
│   ├── bench/            # 基准测试核心模块
│   │   ├── __init__.py  # 包初始化文件
│   │   ├── operator_spec.py  # 算子规格定义和注册系统
│   │   ├── operator_executor.py  # 高性能算子执行器工厂
│   │   ├── pipeline_executor.py  # 高性能管道执行器，支持多算子链式执行
│   │   ├── metrics.py    # 性能指标采集和计算模块
│   │   ├── io.py         # 数据输入输出处理模块
│   │   ├── logger.py     # 统一日志系统，支持灵活的日志级别控制
│   │   ├── cli.py        # 命令行接口，提供run/pipeline/compare等命令
│   │   ├── data_ingest.py  # 数据摄取模块，处理数据加载和预处理
│   │   ├── materialize.py  # 数据物化模块，处理延迟执行的数据
│   │   └── ray_metrics.py  # Ray特定指标采集模块
│   └── engines/          # 引擎实现模块
│       ├── __init__.py  # 包初始化文件
│       ├── spark/       # Spark MLlib引擎实现
│       │   ├── __init__.py  # 包初始化文件
│       │   ├── session.py  # Spark会话管理，创建和配置SparkSession
│       │   └── operators/  # Spark算子实现目录
│       │       ├── __init__.py  # 算子注册模块
│       │       ├── hashingtf.py  # HashingTF特征哈希算子
│       │       ├── idf.py  # 逆文档频率(IDF)算子
│       │       ├── imputer.py  # 缺失值填充算子
│       │       ├── minmaxscaler.py  # 最小-最大标准化算子
│       │       ├── onehotencoder.py  # 独热编码算子
│       │       ├── standardscaler.py  # 标准化算子
│       │       ├── stringindexer.py  # 字符串索引算子
│       │       └── tokenizer.py  # 文本分词算子
│       └── ray/        # Ray引擎实现
│           ├── __init__.py  # 包初始化文件
│           ├── runtime.py  # Ray运行时管理，初始化Ray集群
│           └── operators/  # Ray算子实现目录
│               ├── __init__.py  # 算子注册模块
│               ├── hashingtf.py  # HashingTF特征哈希算子(Ray实现)
│               ├── idf.py  # 逆文档频率(IDF)算子(Ray实现)
│               ├── imputer.py  # 缺失值填充算子(Ray实现)
│               ├── minmaxscaler.py  # 最小-最大标准化算子(Ray实现)
│               ├── onehotencoder.py  # 独热编码算子(Ray实现)
│               ├── standardscaler.py  # 标准化算子(Ray实现)
│               ├── stringindexer.py  # 字符串索引算子(Ray实现)
│               └── tokenizer.py  # 文本分词算子(Ray实现)
├── tests/                # 测试代码目录
│   ├── conftest.py       # PyTest配置文件，定义测试夹具和共享设置
│   ├── test_benchmark_trigger.py  # 基准测试触发机制测试
│   ├── test_operator_contracts.py  # 算子契约测试，验证Spark和Ray输出一致性
│   ├── test_performance_accuracy.py  # 性能准确性测试
│   ├── test_pyspark_op_1.py  # PySpark特定算子测试
│   └── test_smoke.py     # 冒烟测试，验证基本功能和项目结构
```



## 研究目的

比较Ray Data和Spark MLlib中的机器学习预处理算子。



## 研究内容

通过对比分析Ray Data与Spark MLlib在机器学习数据预处理（如标准化、归一化、特 征编码等）算子上的实现机制与性能差异，探究Ray和Spark的差异。



## 实验

### 实验环境

#### 一、硬件配置

* ##### 集群节点

  | 节点名称 | 角色       | CPU核心 | 内存 | 存储 | 网络带宽 |
  | :------- | :--------- | :------ | :--- | :--- | :------- |
  | ecnu01   | Master节点 | 4核     | 16GB | 40GB | 100Mbps  |
  | ecnu02   | Worker节点 | 4核     | 16GB | 40GB | 100Mbps  |
  | ecnu03   | Worker节点 | 4核     | 16GB | 40GB | 100Mbps  |



#### 二、软件环境

* #####  系统与基础软件

| 软件组件   | 版本            |
| :--------- | :-------------- |
| 操作系统   | Ubuntu 24.04    |
| Java环境   | OpenJDK 17.0.17 |
| Python环境 | Python 3.10.17  |

* ##### 大数据框架

| 框架名称     | 版本   | 部署节点                             | 服务端口  |
| :----------- | :----- | :----------------------------------- | :-------- |
| Hadoop HDFS  | 3.3.6  | ecnu01(NameNode) ecnu02/03(DataNode) | 9870 |
| Apache Spark | 3.5.7  | ecnu01(Master) ecnu02/03(Worker)     | 8080 |
| Ray          | 2.52.1 | ecnu01(Head) ecnu02/03(Worker)       | 8265 |



### 实验负载

详细描述使用的数据集和工作负载。



### 实验步骤

列出执行实验的关键步骤，并对关键步骤进行截图，如 MapReduce / Spark / Flink 部署成功后的进程信息、作业执行成功的信息等，**截图能够通过显示用户账号等个性化信息佐证实验的真实性**。



### 实验结果与分析

使用表格和图表直观呈现结果，并解释结果背后的原因。



### 结论

总结研究的主要发现。



### 分工

尽可能详细地写出每个人的具体工作和贡献度，并按贡献度大小进行排序。
