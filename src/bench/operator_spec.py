"""
算子规格定义

定义机器学习预处理算子的统一规格，包括输入输出Schema、参数等。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class OperatorSpec:
    """
    算子规格定义
    
    遵循设计原则：
    - stateful: 是否为有状态算子（需要fit/transform）
    - alignment_policy: 参数对齐策略（Intersection/Ray-Extended/Engine-Specific）
    - output_schema: 输出格式规范（dtype, format等）
    - ray_impl_hint: Ray实现提示（preferred方法、batch_format等）
    """
    name: str
    input_cols: List[str]
    output_cols: List[str]
    params: Dict[str, Any]
    description: str
    engine_impl_names: Optional[Dict[str, str]] = None  # 引擎特定的实现名称映射 {engine: impl_name}
    stateful: bool = True  # 是否为有状态算子（需要fit/transform）
    alignment_policy: str = "Ray-Extended"  # Intersection | Ray-Extended | Engine-Specific
    output_schema: Optional[Dict[str, Any]] = None  # 输出格式规范
    ray_impl_hint: Optional[Dict[str, Any]] = None  # Ray实现提示

    def __post_init__(self):
        """验证规格的合理性"""
        if not self.name:
            raise ValueError("算子名称不能为空")

        if not self.input_cols:
            raise ValueError("输入列不能为空")

        if not self.output_cols:
            raise ValueError("输出列不能为空")

    def get_engine_impl_name(self, engine: str) -> Optional[str]:
        """获取指定引擎的实现名称"""
        if self.engine_impl_names and engine in self.engine_impl_names:
            return self.engine_impl_names[engine]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'params': self.params,
            'description': self.description,
            'engine_impl_names': self.engine_impl_names,
            'stateful': self.stateful,
            'alignment_policy': self.alignment_policy,
            'output_schema': self.output_schema,
            'ray_impl_hint': self.ray_impl_hint
        }


# 算子规格注册表
_OPERATOR_SPECS: Dict[str, OperatorSpec] = {}


def register_operator_spec(spec: OperatorSpec) -> None:
    """注册算子规格"""
    if spec.name in _OPERATOR_SPECS:
        raise ValueError(f"算子 '{spec.name}' 已经注册")

    _OPERATOR_SPECS[spec.name] = spec


def get_operator_spec(name: str) -> OperatorSpec:
    """获取算子规格"""
    if name not in _OPERATOR_SPECS:
        available = list(_OPERATOR_SPECS.keys())
        raise ValueError(f"未知算子 '{name}'，可用算子: {available}")

    return _OPERATOR_SPECS[name]


def get_default_specs() -> Dict[str, OperatorSpec]:
    """获取所有已注册的算子规格"""
    return _OPERATOR_SPECS.copy()


def list_operator_names() -> List[str]:
    """获取所有算子名称"""
    return list(_OPERATOR_SPECS.keys())


# 注册默认算子规格

# StandardScaler - 标准化算子
register_operator_spec(OperatorSpec(
    name="StandardScaler",
    input_cols=["x1", "x2"],  # 默认输入列，可通过参数覆盖
    output_cols=["x1_scaled", "x2_scaled"],
    params={
        "with_mean": True,
        "with_std": True,
        "input_cols": ["x1", "x2"],  # 运行时可覆盖
        "output_cols": ["x1_scaled", "x2_scaled"]
    },
    description="标准化数值特征：移除均值并缩放到单位方差",
    engine_impl_names={
        "spark": "StandardScaler",  # Spark MLlib中的类名
        "ray": "standard_scaler"    # Ray中的实现标识（可自定义）
    },
    stateful=True,  # 有状态算子，需要全局fit
    alignment_policy="Ray-Extended",  # Ray侧自研fit/transform以支持with_mean/with_std开关
    output_schema={
        "dtype": "float64",
        "format": "scalar"  # scalar | dense_vector | sparse_vector
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 优先使用自研fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# 预留其他算子的规格定义位置
# TODO: 添加更多算子规格
# - StringIndexer
# - OneHotEncoder
# - MinMaxScaler
# - Imputer
# - Tokenizer
# - HashingTF
# - IDF

# MinMaxScaler - 最小最大标准化算子
register_operator_spec(OperatorSpec(
    name="MinMaxScaler",
    input_cols=["x1", "x2"],  # 默认输入列，可通过参数覆盖
    output_cols=["x1_scaled", "x2_scaled"],
    params={
        "min": 0.0,
        "max": 1.0,
        "input_cols": ["x1", "x2"],  # 运行时可覆盖
        "output_cols": ["x1_scaled", "x2_scaled"]
    },
    description="最小最大标准化：将数值特征缩放到指定范围[min, max]",
    engine_impl_names={
        "spark": "MinMaxScaler",  # Spark MLlib中的类名
        "ray": "minmax_scaler"    # Ray中的实现标识
    },
    stateful=True,  # 有状态算子，需要全局min/max
    alignment_policy="Ray-Extended",  # Ray侧自研fit/transform以支持任意min/max范围
    output_schema={
        "dtype": "float64",
        "format": "scalar"  # scalar | dense_vector | sparse_vector
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 优先使用自研fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# StringIndexer - 字符串索引算子
register_operator_spec(OperatorSpec(
    name="StringIndexer",
    input_cols=["cat"],  # 默认输入列，可通过参数覆盖
    output_cols=["cat_indexed"],
    params={
        "handle_invalid": "error",  # error | skip | keep
        "input_cols": ["cat"],  # 运行时可覆盖
        "output_cols": ["cat_indexed"]
    },
    description="字符串索引：将类别字符串映射为数字索引",
    engine_impl_names={
        "spark": "StringIndexer",  # Spark MLlib中的类名
        "ray": "string_indexer"    # Ray中的实现标识
    },
    stateful=True,  # 有状态算子，需要全局类别集合
    alignment_policy="Ray-Extended",  # Ray侧自研fit/transform以支持handle_invalid策略
    output_schema={
        "dtype": "int64",
        "format": "scalar"  # scalar | dense_vector | sparse_vector
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 优先使用自研fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# OneHotEncoder - 独热编码算子
register_operator_spec(OperatorSpec(
    name="OneHotEncoder",
    input_cols=["cat_indexed"],  # 默认输入列（通常是StringIndexer的输出）
    output_cols=["cat_onehot"],
    params={
        "drop_last": True,  # 是否丢弃最后一维避免多重共线性
        "handle_invalid": "error",  # error | keep（复用StringIndexer策略）
        "input_cols": ["cat_indexed"],  # 运行时可覆盖
        "output_cols": ["cat_onehot"]
    },
    description="独热编码：将数字索引转换为独热向量",
    engine_impl_names={
        "spark": "OneHotEncoder",  # Spark MLlib中的类名
        "ray": "onehot_encoder"    # Ray中的实现标识
    },
    stateful=True,  # 有状态算子，需要已知类别数量
    alignment_policy="Ray-Extended",  # Ray侧自研以支持drop_last和handle_invalid
    output_schema={
        "dtype": "float64",
        "format": "dense_vector"  # 稠密向量格式
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 优先使用自研fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# Imputer - 缺失值填充算子
register_operator_spec(OperatorSpec(
    name="Imputer",
    input_cols=["x1", "x2"],  # 默认输入列，可通过参数覆盖
    output_cols=["x1_imputed", "x2_imputed"],
    params={
        "strategy": "mean",  # mean | median | mode (Spark支持的策略)
        "input_cols": ["x1", "x2"],  # 运行时可覆盖
        "output_cols": ["x1_imputed", "x2_imputed"]
    },
    description="缺失值填充：使用指定的策略(mean/median/mode)填充缺失值",
    engine_impl_names={
        "spark": "Imputer",  # Spark MLlib中的类名
        "ray": "imputer"    # Ray中的实现标识
    },
    stateful=True,  # 有状态算子，需要全局统计
    alignment_policy="Ray-Extended",  # Ray侧自研以支持median策略
    output_schema={
        "dtype": "float64",
        "format": "scalar"  # scalar | dense_vector | sparse_vector
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 优先使用自研fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# Tokenizer - 文本分词算子
register_operator_spec(OperatorSpec(
    name="Tokenizer",
    input_cols=["text"],  # 默认输入列，可通过参数覆盖
    output_cols=["tokens"],
    params={
        "pattern": None,  # 分词模式（正则表达式），None表示使用默认分词
        "input_col": "text",  # 运行时可覆盖
        "output_col": "tokens"
    },
    description="文本分词：将文本字符串分割为token数组",
    engine_impl_names={
        "spark": "Tokenizer",  # Spark MLlib中的类名
        "ray": "tokenizer"    # Ray中的实现标识
    },
    stateful=False,  # 无状态算子，逐行处理
    alignment_policy="Intersection",  # 使用标准分词行为
    output_schema={
        "dtype": "string",
        "format": "array"  # 字符串数组
    },
    ray_impl_hint={
        "preferred": "map_batches",  # 直接使用map_batches
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# HashingTF - 特征哈希向量化算子
register_operator_spec(OperatorSpec(
    name="HashingTF",
    input_cols=["tokens"],  # 默认输入列，可通过参数覆盖
    output_cols=["tf_features"],
    params={
        "num_features": 2**18,  # 特征维度（默认2^18）
        "input_col": "tokens",  # 运行时可覆盖
        "output_col": "tf_features"
    },
    description="特征哈希向量化：将token数组转换为固定长度向量",
    engine_impl_names={
        "spark": "HashingTF",  # Spark MLlib中的类名
        "ray": "hashing_tf"    # Ray中的实现标识
    },
    stateful=False,  # 无状态算子，逐行哈希
    alignment_policy="Ray-Extended",  # Ray侧自研以固定哈希行为
    output_schema={
        "dtype": "float64",
        "format": "sparse_vector"  # 稀疏向量
    },
    ray_impl_hint={
        "preferred": "map_batches",  # 直接使用map_batches
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))

# IDF - 逆文档频率算子
register_operator_spec(OperatorSpec(
    name="IDF",
    input_cols=["tf_features"],  # 默认输入列，可通过参数覆盖
    output_cols=["tfidf_features"],
    params={
        "min_doc_freq": 1,  # 最小文档频率
        "input_col": "tf_features",  # 运行时可覆盖
        "output_col": "tfidf_features"
    },
    description="逆文档频率：将TF向量转换为TF-IDF向量",
    engine_impl_names={
        "spark": "IDF",  # Spark MLlib中的类名
        "ray": "idf"    # Ray中的实现标识
    },
    stateful=True,  # 有状态算子，需要全局文档频率统计
    alignment_policy="Ray-Extended",  # Ray侧自研fit/transform
    output_schema={
        "dtype": "float64",
        "format": "sparse_vector"  # 稀疏向量
    },
    ray_impl_hint={
        "preferred": "fit_transform",  # 必须使用fit/transform
        "batch_format": "pandas",  # 固定batch_format
        "no_inplace_mutation": True  # 禁止就地修改
    }
))
