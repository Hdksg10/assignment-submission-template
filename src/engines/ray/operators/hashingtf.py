from bench.operator_spec import OperatorSpec
import numpy as np
from scipy import sparse
import hashlib


def _hash_token(token, num_features: int) -> int:
    """稳定哈希到[0, num_features)。使用 md5 保证跨进程一致性。"""
    try:
        if token is None:
            return 0
        h = hashlib.md5(str(token).encode("utf8")).hexdigest()
        return int(h, 16) % int(num_features)
    except Exception as e:
        raise RuntimeError(f"Token哈希失败: {e}")


def _tokens_to_sparse(tokens, num_features: int):
    """把单行 tokens 列表转换为 1 x num_features 的 csr_matrix（计数TF）。"""
    try:
        if tokens is None:
            return sparse.csr_matrix((1, int(num_features)))
        # 兼容字符串/列表/其他类型
        if isinstance(tokens, str):
            tokens = tokens.split()
        counts = {}
        for t in tokens:
            idx = _hash_token(t, num_features)
            counts[idx] = counts.get(idx, 0) + 1.0
        if not counts:
            return sparse.csr_matrix((1, int(num_features)))
        cols = np.fromiter(counts.keys(), dtype=np.int32)
        data = np.fromiter(counts.values(), dtype=np.float64)
        rows = np.zeros_like(cols)
        return sparse.csr_matrix((data, (rows, cols)), shape=(1, int(num_features)))
    except Exception as e:
        raise RuntimeError(f"Tokens转换为稀疏矩阵失败: {e}")


def run_hashingtf_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用 Ray map_batches 实现 HashingTF：
    - 将 tokens 列（或多列）哈希到固定维度，产生稀疏 TF 向量 (scipy.sparse.csr_matrix)。
    支持 input_cols/output_cols 为字符串或列表；若多列输入，则逐列映射到对应输出列。
    """
    try:
        input_cols = spec.params.get("input_cols", spec.input_cols)
        output_cols = spec.params.get("output_cols", spec.output_cols)
        num_features = spec.params.get("num_features", 2**18)

        # 规范化为列表
        if isinstance(input_cols, str):
            input_cols = [input_cols]
        if output_cols is None:
            output_cols = input_cols
        elif isinstance(output_cols, str):
            output_cols = [output_cols]

        def _hash_batch(
            batch, in_cols=input_cols, out_cols=output_cols, nf=num_features
        ):
            # 期望 batch 为 pandas.DataFrame
            for i, in_col in enumerate(in_cols):
                out_col = out_cols[i] if i < len(out_cols) else in_col
                # 对每行生成稀疏向量
                result = []
                # 使用 itertuples 比较快
                for val in batch[in_col].values:
                    result.append(_tokens_to_sparse(val, nf))
                batch[out_col] = result
            return batch

        processed_dataset = ray_dataset.map_batches(_hash_batch, batch_format="pandas")
        return processed_dataset
    except Exception as e:
        raise RuntimeError(f"HashingTF执行失败: {e}")
