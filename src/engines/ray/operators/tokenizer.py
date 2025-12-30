"""Ray实现Tokenizer算子。"""

from bench.operator_spec import OperatorSpec


def run_tokenizer_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用Ray实现Tokenizer算子。

    参数:
    - ray_dataset: Ray Dataset对象
    - input_cols: 输入列列表
    - output_cols: 输出列列表
    - params: 其他参数（未使用）

    返回:
    - 处理后的Ray Dataset对象
    """
    try:
        from ray.data.preprocessors import Tokenizer

        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        pattern = spec.params.get("pattern", None)

        tokenizer = Tokenizer(
            columns=input_cols,
            output_columns=output_cols,
        )
        processed_dataset = tokenizer.fit_transform(ray_dataset)

        return processed_dataset
    except Exception as e:
        raise RuntimeError(f"Tokenizer执行失败: {e}")
