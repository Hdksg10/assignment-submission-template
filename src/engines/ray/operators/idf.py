# ...existing code...
import numpy as np
from scipy import sparse
from bench.operator_spec import OperatorSpec

try:
    from ...bench.logger import get_logger

    _logger = get_logger(__name__)
except ImportError:
    _logger = None


def run_idf_with_ray_data(ray_dataset, spec: OperatorSpec):
    """
    使用 Ray Data 实现 IDF（fit -> transform），当前实现将数据收集到 pandas 以统计文档频率（适用于中小规模数据集）。
    支持输入 TF 列为 scipy.sparse.csr_matrix 或 numpy.ndarray/list。输出为与输入相同格式的 TF-IDF 向量。
    """
    try:
        input_col = spec.params.get("input_col", spec.input_cols[0])
        output_col = spec.params.get("output_col", spec.output_cols[0])
        min_doc_freq = int(spec.params.get("min_doc_freq", 1))

        # 收集到 pandas 以计算全局文档频率（可替换为分布式聚合）
        df_all = ray_dataset.to_pandas()
        if input_col not in df_all.columns:
            raise ValueError(f"输入列 {input_col} 不存在")

        doc_count = len(df_all)
        if doc_count == 0:
            raise ValueError("没有文档用于计算 IDF")

        # 根据首个非空向量确定 vocab_size（若为稀疏向量则使用 shape）
        vocab_size = None
        for v in df_all[input_col]:
            if v is None:
                continue
            if sparse.issparse(v):
                vocab_size = v.shape[1] if v.ndim == 2 else v.shape[0]
                break
            else:
                arr = np.asarray(v).ravel()
                vocab_size = arr.shape[0]
                break
        if vocab_size is None:
            raise ValueError("无法确定词汇表大小（所有TF向量均为空）")
        df_counts = np.zeros(int(vocab_size), dtype=np.int64)

        # 累加文档频率
        for v in df_all[input_col]:
            if v is None:
                continue
            if sparse.issparse(v):
                vec = v.tocoo()
                cols = np.unique(vec.col) if vec.ndim == 2 else np.unique(vec.indices)
                df_counts[cols] += 1
            else:
                arr = np.asarray(v).ravel()
                nz = np.flatnonzero(arr)
                if nz.size:
                    df_counts[nz] += 1

        # 计算 IDF（平滑）
        idf = np.log((1.0 + doc_count) / (1.0 + df_counts)) + 1.0
        # 对低频词置零
        idf[df_counts < min_doc_freq] = 0.0

        # 定义 batch-level transform
        def _to_tfidf_batch(
            batch, in_col=input_col, out_col=output_col, idf_arr=idf, vocab=vocab_size
        ):
            # 期望 batch 为 pandas.DataFrame
            result_col = []
            for v in batch[in_col].values:
                if v is None:
                    # 生成空向量
                    result_col.append(sparse.csr_matrix((1, int(vocab))))
                    continue
                if sparse.issparse(v):
                    coo = v.tocoo()
                    cols = coo.col if coo.ndim == 2 else coo.indices
                    vals = coo.data
                    # 防止索引越界：对超出 vocab 的索引取模或截断（此处截断）
                    valid_mask = cols < len(idf_arr)
                    if not valid_mask.all():
                        cols = cols[valid_mask]
                        vals = vals[valid_mask]
                    if cols.size == 0:
                        result_col.append(sparse.csr_matrix((1, int(vocab))))
                        continue
                    new_data = vals * idf_arr[cols]
                    new = sparse.csr_matrix(
                        (new_data, (np.zeros_like(cols), cols)), shape=(1, int(vocab))
                    )
                    result_col.append(new)
                else:
                    arr = np.asarray(v).ravel()
                    # 调整长度
                    if arr.shape[0] != vocab:
                        if arr.shape[0] < vocab:
                            tmp = np.zeros(int(vocab), dtype=arr.dtype)
                            tmp[: arr.shape[0]] = arr
                            arr = tmp
                        else:
                            arr = arr[: int(vocab)]
                    result_col.append(arr * idf_arr)
            batch[out_col] = result_col
            return batch

        # 应用 map_batches 转换（保持分布式处理）
        processed = ray_dataset.map_batches(_to_tfidf_batch, batch_format="pandas")
        return processed
    except Exception as e:
        raise RuntimeError(f"IDF执行失败: {e}")


# ...existing code...
