#!/bin/bash
set -euo pipefail

echo "=== Spark 3.5.7 环境设置（单机/伪分布式 Standalone 适用）==="

# ------------------------------
# 配置项
# ------------------------------
SPARK_VERSION="3.5.7"
HADOOP_PROFILE="hadoop3"
SPARK_PKG="spark-${SPARK_VERSION}-bin-${HADOOP_PROFILE}"
SPARK_TGZ="${SPARK_PKG}.tgz"

PRIMARY_URL="https://downloads.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_TGZ}"
ARCHIVE_URL="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_TGZ}"

# 优先装到 /opt（需要 sudo）；没权限则装到 $HOME
INSTALL_BASE="/opt"
if ! (mkdir -p "${INSTALL_BASE}" 2>/dev/null); then
  INSTALL_BASE="${HOME}"
fi
INSTALL_DIR="${INSTALL_BASE}/${SPARK_PKG}"
SPARK_SYMLINK="${INSTALL_BASE}/spark"

# ------------------------------
# 工具函数
# ------------------------------
verlte() { [ "$1" = "$(printf "%s\n%s\n" "$1" "$2" | sort -V | head -n1)" ]; }  # 1 <= 2

get_java_major() {
  local v
  v="$(java -version 2>&1 | head -n 1 | sed -n 's/.*"\(.*\)".*/\1/p')"
  # "1.8.0_372" -> 8, "11.0.22" -> 11, "17.0.10" -> 17
  if [[ "$v" == 1.* ]]; then
    echo "${v#1.}" | cut -d. -f1
  else
    echo "$v" | cut -d. -f1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 1 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "✗ 缺少下载工具：请安装 curl 或 wget"
    echo "  Ubuntu/Debian: sudo apt install -y curl"
    exit 1
  fi
}

# ------------------------------
# 检查 Java
# ------------------------------
echo "检查 Java 环境..."
if command -v java >/dev/null 2>&1; then
  JAVA_VERSION_STR="$(java -version 2>&1 | head -n 1)"
  JAVA_MAJOR="$(get_java_major)"
  echo "✓ ${JAVA_VERSION_STR}"
  if [[ "$JAVA_MAJOR" != "8" && "$JAVA_MAJOR" != "11" && "$JAVA_MAJOR" != "17" ]]; then
    echo "✗ 检测到 Java 主版本：${JAVA_MAJOR}（Spark 3.5.7 推荐 Java 8/11/17）"
    echo "  建议安装：sudo apt install -y openjdk-17-jdk"
    exit 1
  fi
else
  echo "✗ Java 未安装（Spark 3.5.7 建议 Java 17，或 11/8）"
  echo "  Ubuntu/Debian: sudo apt install -y openjdk-17-jdk"
  exit 1
fi

# 尝试推断 JAVA_HOME（若未设置）
if [[ -z "${JAVA_HOME:-}" ]]; then
  JAVA_BIN="$(readlink -f "$(command -v java)")"
  export JAVA_HOME="$(dirname "$(dirname "$JAVA_BIN")")"
  echo "✓ 推断 JAVA_HOME: $JAVA_HOME"
fi

# ------------------------------
# 检查 Python
# ------------------------------
echo "检查 Python 环境..."
if command -v python3 >/dev/null 2>&1; then
  PYTHON_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
  echo "✓ Python版本: $PYTHON_VERSION"
  # Spark 3.5.x 需要 Python 3.8+
  PYTHON_MAJOR_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if verlte "3.8" "$PYTHON_MAJOR_MINOR"; then
    : # ok
  else
    echo "✗ Python 版本过低：$PYTHON_VERSION（需要 3.8+）"
    exit 1
  fi
else
  echo "✗ Python3 未安装"
  exit 1
fi

# ------------------------------
# 下载并安装 Spark 3.5.7
# ------------------------------
echo "安装 Spark ${SPARK_VERSION}（${SPARK_PKG}）..."

# 如果已存在安装目录就跳过下载解压
if [[ -d "$INSTALL_DIR" ]]; then
  echo "✓ 已存在安装目录：$INSTALL_DIR"
else
  TMPDIR="$(mktemp -d)"
  TGZ_PATH="${TMPDIR}/${SPARK_TGZ}"

  echo "下载 Spark：$PRIMARY_URL"
  if ! download_file "$PRIMARY_URL" "$TGZ_PATH"; then
    echo "主下载源失败，改用 Archive：$ARCHIVE_URL"
    download_file "$ARCHIVE_URL" "$TGZ_PATH"
  fi

  echo "解压 Spark..."
  tar -xzf "$TGZ_PATH" -C "$TMPDIR"

  # 复制/移动到目标目录：优先 /opt（需要 sudo）
  if [[ "$INSTALL_BASE" == "/opt" ]]; then
    sudo mkdir -p "/opt"
    sudo rm -rf "$INSTALL_DIR"
    sudo mv "${TMPDIR}/${SPARK_PKG}" "$INSTALL_DIR"
  else
    rm -rf "$INSTALL_DIR"
    mv "${TMPDIR}/${SPARK_PKG}" "$INSTALL_DIR"
  fi

  rm -rf "$TMPDIR"
  echo "✓ Spark 安装完成：$INSTALL_DIR"
fi

# 建立 /opt/spark 或 ~/spark 软链接，方便 SPARK_HOME 固定
echo "设置 SPARK_HOME..."
if [[ "$INSTALL_BASE" == "/opt" ]]; then
  sudo ln -sfn "$INSTALL_DIR" "$SPARK_SYMLINK"
else
  ln -sfn "$INSTALL_DIR" "$SPARK_SYMLINK"
fi

export SPARK_HOME="$SPARK_SYMLINK"
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
echo "✓ SPARK_HOME: $SPARK_HOME"

# ------------------------------
# 安装 PySpark（对齐 3.5.7）
# ------------------------------
echo "安装/更新 PySpark==${SPARK_VERSION}（用于 python3 直接 import pyspark）..."
python3 -m pip install -U "pyspark==${SPARK_VERSION}"

# ------------------------------
# 冒烟测试：用 spark-submit 运行一个最小任务
# ------------------------------
echo "测试 Spark 安装（spark-submit 冒烟测试）..."
TEST_PY="$(mktemp /tmp/spark_test_XXXX.py)"
cat > "$TEST_PY" <<'PY'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").getOrCreate()
print("Spark版本:", spark.version)
spark.range(1, 10).count()
spark.stop()
print("✓ Spark 冒烟测试成功")
PY

"$SPARK_HOME/bin/spark-submit" "$TEST_PY"
rm -f "$TEST_PY"

echo
echo "✓ Spark 3.5.7 环境设置完成！"
echo "下一步（单机伪分布式 Standalone）你可以运行："
echo "  start-master.sh"
echo "  start-worker.sh spark://127.0.0.1:7077"
