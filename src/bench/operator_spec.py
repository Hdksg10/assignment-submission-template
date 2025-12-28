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
