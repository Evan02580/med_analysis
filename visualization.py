import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pandas as pd


# 参数设置
FILE_PATH = "datasets/medical_data.csv"
DROPPED_COL = ["Number", "JOA", "NRS/VAS"]
LABEL_COL = "dropped"


def read_data(csv_path, standardize=False):
    """数据预处理，返回特征矩阵X，标签y，类别列和数值列列表"""
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # 标准化列名, 去除 DROPPED_COL 中的列
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=DROPPED_COL)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 处理分类变量
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            df[col] = LabelEncoder().fit_transform(df[col])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, df.columns.tolist()


def plot_feature_distribution(df, feature_cols, label_col):
    """绘制特征分布图"""
    for col in feature_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue=label_col, kde=True, stat="density", common_norm=False)
        plt.title(f'Distribution of {col} by {label_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(title=label_col)
        plt.show()


# 绘制特征分布图示例
if __name__ == "__main__":
    df, cols = read_data(FILE_PATH, standardize=False)
    feature_cols = [col for col in cols if col != LABEL_COL]
    plot_feature_distribution(df, feature_cols, LABEL_COL)
