#!/bin/bash
# 数据集下载脚本（优先国内镜像/代理，失败自动回退）
set -euo pipefail
IFS=$'\n\t'

DATASET_NAME=${1:-"house_prices"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PROJECT_ROOT/data/large}"
mkdir -p "$DOWNLOAD_DIR"
export DOWNLOAD_DIR   # 让内嵌 python3 能读到 os.environ["DOWNLOAD_DIR"]

# -----------------------------
# 国内镜像/代理配置
# -----------------------------
# GitHub Raw / Releases / Archive 代理（按顺序尝试）
GH_MIRRORS=(
  "https://mirror.ghproxy.com/"
  "https://gh-proxy.com/"
  "https://cf.ghproxy.cc/"
)

# Hugging Face 国内镜像（可通过环境变量覆盖）
HF_ENDPOINT_DEFAULT="https://hf-mirror.com"

# Python 包安装的国内 PyPI 镜像（可通过环境变量覆盖）
PIP_MIRROR_DEFAULT="https://pypi.tuna.tsinghua.edu.cn/simple"

# Hugging Face 直连连通性测试 URL（可覆盖）
HF_TEST_URL_DEFAULT="https://huggingface.co/api/datasets/ag_news"

# -----------------------------
# 通用下载函数：多 URL fallback
# -----------------------------
download_with_fallback() {
  local output="$1"; shift
  local -a urls=("$@")
  local tmp="${output}.tmp"

  rm -f "$tmp"
  for u in "${urls[@]}"; do
    echo "  -> 尝试: $u"
    if curl -fL --retry 3 --retry-all-errors --connect-timeout 10 -o "$tmp" "$u"; then
      mv "$tmp" "$output"
      return 0
    fi
  done

  rm -f "$tmp"
  return 1
}

# -----------------------------
# Hugging Face endpoint 自动选择：
# - 若能直连 huggingface.co：不使用镜像（unset HF_ENDPOINT）
# - 否则：使用镜像（HF_ENDPOINT_DEFAULT 或用户自定义 HF_ENDPOINT）
# -----------------------------
setup_hf_endpoint() {
  local test_url="${HF_TEST_URL:-$HF_TEST_URL_DEFAULT}"

  echo "检查 Hugging Face 直连连通性: $test_url"
  if curl -fsSL --connect-timeout 5 --max-time 10 "$test_url" >/dev/null 2>&1; then
    echo "✓ Hugging Face 直连可用：使用官方 endpoint（不使用镜像）"
    unset HF_ENDPOINT
  else
    export HF_ENDPOINT="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"
    echo "⚠ Hugging Face 直连不可用：使用镜像/代理 HF_ENDPOINT=$HF_ENDPOINT"
  fi
}

echo "=== 下载数据集: $DATASET_NAME ==="
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"

case "$DATASET_NAME" in
  "house_prices")
    echo "下载房价预测数据集（优先 GitHub 国内代理）..."
    ORIGIN_URL="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    OUTPUT_FILE="$DOWNLOAD_DIR/house_prices.csv"

    URLS=()
    for m in "${GH_MIRRORS[@]}"; do
      URLS+=("${m}${ORIGIN_URL}")
    done
    URLS+=("${ORIGIN_URL}") # 最后回退官方直连

    if download_with_fallback "$OUTPUT_FILE" "${URLS[@]}"; then
      echo "✓ 下载成功: $OUTPUT_FILE"
      wc -l "$OUTPUT_FILE" || true
      ls -lh "$OUTPUT_FILE" || true
    else
      echo "✗ 下载失败：国内代理与官方直连均不可用"
      exit 1
    fi
    ;;

  "20newsgroups")
    echo "下载20新闻组数据集（自动判断直连/镜像）..."
    setup_hf_endpoint
    export PIP_MIRROR="${PIP_MIRROR:-$PIP_MIRROR_DEFAULT}"
    export HF_HOME="${HF_HOME:-$DOWNLOAD_DIR/.hf_cache}"
    mkdir -p "$HF_HOME"

    python3 - <<'PY'
import os, sys, importlib, subprocess

download_dir = os.environ["DOWNLOAD_DIR"]
pip_mirror = os.environ.get("PIP_MIRROR", "")
hf_endpoint = os.environ.get("HF_ENDPOINT", "")

def ensure(pkg: str):
    try:
        importlib.import_module(pkg.split("==")[0])
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--user"]
        if pip_mirror:
            cmd += ["-i", pip_mirror]
        cmd += [pkg]
        print(f"[install] {' '.join(cmd)}")
        subprocess.check_call(cmd)

ensure("pandas")
ensure("datasets==3.4.0")

import pandas as pd
from datasets import load_dataset, concatenate_datasets

print(f"HF_ENDPOINT={'<official:https://huggingface.co>' if not hf_endpoint else hf_endpoint}")
ds = load_dataset("SetFit/20_newsgroups")  # train/test
splits = [ds[k] for k in ds.keys()]
all_ds = concatenate_datasets(splits)

df = all_ds.to_pandas()

# 兼容不同字段命名
text_col = "text" if "text" in df.columns else ("sentence" if "sentence" in df.columns else None)
label_col = "label" if "label" in df.columns else ("category" if "category" in df.columns else None)
if text_col is None or label_col is None:
    raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

out_df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "category"})
output_file = os.path.join(download_dir, "20newsgroups.csv")
out_df.to_csv(output_file, index=False)

print(f"✓ 保存到: {output_file}")
print(f"形状: {out_df.shape}")
PY
    ;;

  "ag_news")
    echo "下载 AG News 数据集（自动判断直连/镜像）..."
    setup_hf_endpoint
    export PIP_MIRROR="${PIP_MIRROR:-$PIP_MIRROR_DEFAULT}"
    export HF_HOME="${HF_HOME:-$DOWNLOAD_DIR/.hf_cache}"
    mkdir -p "$HF_HOME"

    python3 - <<'PY'
import os, sys, importlib, subprocess

download_dir = os.environ["DOWNLOAD_DIR"]
pip_mirror = os.environ.get("PIP_MIRROR", "")
hf_endpoint = os.environ.get("HF_ENDPOINT", "")

def ensure(pkg: str):
    try:
        importlib.import_module(pkg.split("==")[0])
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--user"]
        if pip_mirror:
            cmd += ["-i", pip_mirror]
        cmd += [pkg]
        print(f"[install] {' '.join(cmd)}")
        subprocess.check_call(cmd)

ensure("pandas")
ensure("datasets==3.4.0")

import pandas as pd
from datasets import load_dataset, concatenate_datasets

print(f"HF_ENDPOINT={'<official:https://huggingface.co>' if not hf_endpoint else hf_endpoint}")
ds = load_dataset("ag_news")  # DatasetDict: train/test

splits = [ds[k] for k in ds.keys()]
all_ds = concatenate_datasets(splits)

df = all_ds.to_pandas()

# 期望列：text / label
if "text" not in df.columns or "label" not in df.columns:
    raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

out_df = df[["text", "label"]]
output_file = os.path.join(download_dir, "ag_news.csv")
out_df.to_csv(output_file, index=False)

print(f"✓ 保存到: {output_file}")
print(f"形状: {out_df.shape}")
PY
    ;;

  "yelp_polarity")
    echo "下载 Yelp Polarity 数据集（自动判断直连/镜像）..."
    setup_hf_endpoint
    export PIP_MIRROR="${PIP_MIRROR:-$PIP_MIRROR_DEFAULT}"
    export HF_HOME="${HF_HOME:-$DOWNLOAD_DIR/.hf_cache}"
    mkdir -p "$HF_HOME"

    python3 - <<'PY'
import os, sys, importlib, subprocess

download_dir = os.environ["DOWNLOAD_DIR"]
pip_mirror = os.environ.get("PIP_MIRROR", "")
hf_endpoint = os.environ.get("HF_ENDPOINT", "")

def ensure(pkg: str):
    try:
        importlib.import_module(pkg.split("==")[0])
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--user"]
        if pip_mirror:
            cmd += ["-i", pip_mirror]
        cmd += [pkg]
        print(f"[install] {' '.join(cmd)}")
        subprocess.check_call(cmd)

ensure("pandas")
ensure("datasets==3.4.0")

import pandas as pd
from datasets import load_dataset, concatenate_datasets

print(f"HF_ENDPOINT={'<official:https://huggingface.co>' if not hf_endpoint else hf_endpoint}")
ds = load_dataset("fancyzhx/yelp_polarity")  # DatasetDict: train/test

splits = [ds[k] for k in ds.keys()]
all_ds = concatenate_datasets(splits)

df = all_ds.to_pandas()

# 期望列：text / label
if "text" not in df.columns or "label" not in df.columns:
    raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

out_df = df[["text", "label"]]
output_file = os.path.join(download_dir, "yelp_polarity.csv")
out_df.to_csv(output_file, index=False)

print(f"✓ 保存到: {output_file}")
print(f"形状: {out_df.shape}")
PY
    ;;

  "credit_fraud")
    echo "下载信用卡欺诈检测数据集（自动判断直连/镜像，避免 Kaggle 认证）..."
    echo "注意: 数据集较大（~150MB），可能需要几分钟"
    setup_hf_endpoint
    export PIP_MIRROR="${PIP_MIRROR:-$PIP_MIRROR_DEFAULT}"
    export HF_HOME="${HF_HOME:-$DOWNLOAD_DIR/.hf_cache}"
    mkdir -p "$HF_HOME"

    python3 - <<'PY'
import os, sys, importlib, subprocess

download_dir = os.environ["DOWNLOAD_DIR"]
pip_mirror = os.environ.get("PIP_MIRROR", "")
hf_endpoint = os.environ.get("HF_ENDPOINT", "")

def ensure(pkg: str):
    try:
        importlib.import_module(pkg.split("==")[0])
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--user"]
        if pip_mirror:
            cmd += ["-i", pip_mirror]
        cmd += [pkg]
        print(f"[install] {' '.join(cmd)}")
        subprocess.check_call(cmd)

ensure("pandas")
ensure("datasets==3.4.0")

import pandas as pd
from datasets import load_dataset, concatenate_datasets

print(f"HF_ENDPOINT={'<official:https://huggingface.co>' if not hf_endpoint else hf_endpoint}")
ds = load_dataset("David-Egea/Creditcard-fraud-detection")

# 常见为单 split（train），这里统一拼起来更稳
splits = [ds[k] for k in ds.keys()]
all_ds = concatenate_datasets(splits)

df = all_ds.to_pandas()
output_file = os.path.join(download_dir, "credit_fraud.csv")
df.to_csv(output_file, index=False)

print(f"✓ 保存到: {output_file}")
print(f"形状: {df.shape}")
PY
    ;;

  *)
    echo "未知数据集: $DATASET_NAME"
    echo "可用数据集:"
    echo "  house_prices   - 房价预测数据集（GitHub 国内代理）"
    echo "  20newsgroups   - 20新闻组文本数据集（HF 直连/镜像自动选择）"
    echo "  ag_news        - AG News 文本分类数据集（HF 直连/镜像自动选择）"
    echo "  yelp_polarity  - Yelp Polarity 情感分类数据集（HF 直连/镜像自动选择）"
    echo "  credit_fraud   - 信用卡欺诈检测数据集（HF 直连/镜像自动选择）"
    exit 1
    ;;
esac

echo
echo "数据集下载完成！"
echo "文件位置: $DOWNLOAD_DIR"
echo

echo "验证数据集:"
case "$DATASET_NAME" in
  "house_prices")
    echo "校验house_prices数据集..."
    python3 - <<'PY'
import pandas as pd, os
path = os.path.join(os.environ["DOWNLOAD_DIR"], "house_prices.csv")
df = pd.read_csv(path)
print(f"文件: {path}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("列名:", list(df.columns))
print("数据类型:")
print(df.dtypes)
PY
    ;;

  "20newsgroups")
    echo "校验20newsgroups数据集..."
    python3 - <<'PY'
import pandas as pd, os
path = os.path.join(os.environ["DOWNLOAD_DIR"], "20newsgroups.csv")
df = pd.read_csv(path)
print(f"文件: {path}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("类别数量:", df["category"].nunique())
print("样本文本长度分布:")
print(df["text"].astype(str).str.len().describe())
PY
    ;;

  "ag_news")
    echo "校验ag_news数据集..."
    python3 - <<'PY'
import pandas as pd, os
path = os.path.join(os.environ["DOWNLOAD_DIR"], "ag_news.csv")
df = pd.read_csv(path)
print(f"文件: {path}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("label 数量:", df["label"].nunique())
print("样本文本长度分布:")
print(df["text"].astype(str).str.len().describe())
PY
    ;;

  "yelp_polarity")
    echo "校验yelp_polarity数据集..."
    python3 - <<'PY'
import pandas as pd, os
path = os.path.join(os.environ["DOWNLOAD_DIR"], "yelp_polarity.csv")
df = pd.read_csv(path)
print(f"文件: {path}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("label 数量:", df["label"].nunique())
print("样本文本长度分布:")
print(df["text"].astype(str).str.len().describe())
PY
    ;;

  "credit_fraud")
    echo "校验credit_fraud数据集..."
    python3 - <<'PY'
import pandas as pd, os
path = os.path.join(os.environ["DOWNLOAD_DIR"], "credit_fraud.csv")
df = pd.read_csv(path)
print(f"文件: {path}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("前10列:", list(df.columns[:10]))
if "Class" in df.columns:
    print("Class 分布:")
    print(df["Class"].value_counts(dropna=False).head())
PY
    ;;
esac
