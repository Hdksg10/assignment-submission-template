import pandas as pd
from bench.operator_spec import OperatorSpec
from ray.data.preprocessors import SimpleImputer

try:
    from ...bench.logger import get_logger

    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_imputer_with_ray_data(ray_dataset, spec: OperatorSpec):
    """使用Ray Data运行Imputer算子。

    参数:
        ray_dataset (ray.data.Dataset): 输入的Ray Data数据集。
        spec (OperatorSpec): 包含Imputer参数的算子规范。
    返回:
        ray.data.Dataset: 处理后的Ray Data数据集。
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        strategy = spec.params.get("strategy", "mean")

        if strategy == "median":
            # 计算全局中位数（转为 pandas 可能比较重，适用于中小数据集）
            df_all = ray_dataset.to_pandas()
            medians = df_all[input_cols].median(numeric_only=True).to_dict()

            def _fill_with_median(
                batch, med=medians, in_cols=input_cols, out_cols=output_cols
            ):
                # 对每列应用全局中位数填充，保持列名映射
                for i, col in enumerate(in_cols):
                    out_col = out_cols[i] if i < len(out_cols) else col
                    batch[out_col] = batch[col].fillna(med.get(col, 0))
                return batch

            processed_dataset = ray_dataset.map_batches(
                _fill_with_median, batch_format="pandas"
            )
        elif strategy == "mode":
            imputer = SimpleImputer(
                columns=input_cols,
                output_columns=output_cols,
                strategy="most_frequent",
            )
            processed_dataset = imputer.fit_transform(ray_dataset)
        else:
            imputer = SimpleImputer(
                columns=input_cols,
                output_columns=output_cols,
                strategy=strategy,
            )
            processed_dataset = imputer.fit_transform(ray_dataset)

        return processed_dataset
    except Exception as e:
        raise RuntimeError(f"Imputer执行失败: {e}")
