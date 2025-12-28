#!/bin/bash

# 数据集下载脚本

set -e

DATASET_NAME=${1:-"house_prices"}
DOWNLOAD_DIR="/tmp/benchmark_datasets"
mkdir -p "$DOWNLOAD_DIR"

echo "=== 下载数据集: $DATASET_NAME ==="

case $DATASET_NAME in
    "house_prices")
        echo "下载房价预测数据集..."
        URL="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        OUTPUT_FILE="$DOWNLOAD_DIR/house_prices.csv"

        if curl -s -o "$OUTPUT_FILE" "$URL"; then
            echo "✓ 下载成功: $OUTPUT_FILE"
            wc -l "$OUTPUT_FILE"
        else
            echo "✗ 下载失败"
            exit 1
        fi
        ;;

    "credit_fraud")
        echo "下载信用卡欺诈检测数据集..."
        echo "注意: 此数据集较大，可能需要几分钟"
        URL="https://www.kaggle.com/mlg-ulb/creditcardfraud/download"
        echo "请手动从Kaggle下载: $URL"
        echo "或者使用以下命令:"
        echo "  pip install kaggle"
        echo "  kaggle datasets download -d mlg-ulb/creditcardfraud"
        echo "  unzip creditcardfraud.zip -d $DOWNLOAD_DIR"
        exit 1
        ;;

    "20newsgroups")
        echo "下载20新闻组数据集..."
        python3 -c "
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

print('下载20新闻组数据集...')
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': newsgroups.data, 'category': newsgroups.target})
output_file = '$DOWNLOAD_DIR/20newsgroups.csv'
df.to_csv(output_file, index=False)
print(f'✓ 保存到: {output_file}')
print(f'形状: {df.shape}')
        "
        ;;

    *)
        echo "未知数据集: $DATASET_NAME"
        echo "可用数据集:"
        echo "  house_prices  - 房价预测数据集"
        echo "  credit_fraud  - 信用卡欺诈检测数据集"
        echo "  20newsgroups  - 20新闻组文本数据集"
        exit 1
        ;;
esac

echo
echo "数据集下载完成！"
echo "文件位置: $DOWNLOAD_DIR"
echo
echo "验证数据集:"
case $DATASET_NAME in
    "house_prices")
        echo "校验house_prices数据集..."
        python3 -c "
import pandas as pd
df = pd.read_csv('$DOWNLOAD_DIR/house_prices.csv')
print(f'行数: {len(df)}')
print(f'列数: {len(df.columns)}')
print('列名:', list(df.columns))
print('数据类型:')
print(df.dtypes)
        "
        ;;
    "20newsgroups")
        echo "校验20newsgroups数据集..."
        python3 -c "
import pandas as pd
df = pd.read_csv('$DOWNLOAD_DIR/20newsgroups.csv')
print(f'行数: {len(df)}')
print(f'列数: {len(df.columns)}')
print('类别数量:', df['category'].nunique())
print('样本文本长度分布:')
print(df['text'].str.len().describe())
        "
        ;;
esac
