import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# 参数设置
FILE_PATH = "datasets/filled_data_v4.csv"
# KEEP_COL = ["length","椎间盘退化","segment","是否脱出/游离","狭窄分级","MSU分级","MSU分区","椎管骨性直径","椎管最小直径","椎管面积"]
KEEP_COL = ["length"]
LABEL_COL = "dropped"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def read_data(csv_path, standardize=True):
    """数据预处理，返回特征矩阵X，标签y，类别列和数值列列表"""
    df = pd.read_csv(csv_path)

    # 标准化列名, 只保留 KEEP_COL 中的列
    df.columns = [str(c).strip() for c in df.columns]
    print(df.columns)
    df = df.loc[:, KEEP_COL + [LABEL_COL]]
    print(df.columns)
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan)

    # 处理分类变量
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            df[col] = LabelEncoder().fit_transform(df[col])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 分离特征和标签
    features = df.drop(columns=[LABEL_COL])
    labels = df[LABEL_COL]
    if standardize:
        features = StandardScaler().fit_transform(features)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        shuffle=True
    )
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples ({(1 - TEST_SIZE) * 10:.0f}:{TEST_SIZE * 10:.0f})")
    print(f"Weight of positive samples in training set: {np.mean(y_train):.3f}")
    print(f"Weight of positive samples in testing set: {np.mean(y_test):.3f}")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """训练逻辑回归分类器"""
    print("|   LR   |", end="")
    lr_clf = LogisticRegression(
        max_iter=100, solver="liblinear", random_state=RANDOM_STATE
    )
    lr_clf.fit(X_train, y_train)
    return lr_clf


def train_decision_tree(X_train, y_train):
    """训练决策树分类器"""
    print("|   DT   |", end="")
    dt_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_clf.fit(X_train, y_train)
    return dt_clf


def train_random_forest(X_train, y_train):
    """训练随机森林分类器"""
    print("|   RF   |", end="")
    rf_clf = RandomForestClassifier(n_estimators=5, random_state=RANDOM_STATE)
    rf_clf.fit(X_train, y_train)
    return rf_clf


def train_xgboost(X_train, y_train):
    """训练XGBoost分类器"""
    print("| XGBoost|", end="")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
    specificity = np.sum((y_test == 0) & (y_pred == 0)) / np.sum(y_test == 0)
    f1 = f1_score(y_test, y_pred)
    PPV = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    NPV = np.sum((y_test == 0) & (y_pred == 0)) / np.sum(y_pred == 0)

    print(f" {auc:.3f} | {accuracy:.3f} | {sensitivity:.3f} | {specificity:.3f} | {f1:.3f} | {PPV:.3f} | {NPV:.3f} |")


def plot_roc_curves(models, model_names, X_train, y_train, X_test, y_test, save_path="results/roc_curves.png"):
    """绘制所有模型在训练集和测试集上的ROC曲线

    Args:
        models: 模型列表
        model_names: 模型名称列表
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        save_path: 保存图片的路径
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四种颜色

    # 创建两个子图：训练集和测试集
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制训练集ROC曲线
    ax_train = axes[0]
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_proba_train = model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
        auc_train = roc_auc_score(y_train, y_proba_train)
        ax_train.plot(fpr_train, tpr_train, color=colors[idx], lw=2,
                     label=f'{name} (AUC = {auc_train:.3f})')

    # 绘制对角线 y=x
    ax_train.plot([0, 1], [0, 1], 'k--', lw=1, label='Reference')
    ax_train.set_xlabel('1 - Specificity', fontsize=12)
    ax_train.set_ylabel('Sensitivity', fontsize=12)
    ax_train.set_title('ROC Curve - Training Set', fontsize=14, fontweight='bold')
    ax_train.legend(loc='lower right', fontsize=10)
    ax_train.grid(True, alpha=0.3)
    ax_train.set_xlim([0.0, 1.0])
    ax_train.set_ylim([0.0, 1.05])

    # 绘制测试集ROC曲线
    ax_test = axes[1]
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_proba_test = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
        auc_test = roc_auc_score(y_test, y_proba_test)
        ax_test.plot(fpr_test, tpr_test, color=colors[idx], lw=2,
                    label=f'{name} (AUC = {auc_test:.3f})')

    # 绘制对角线 y=x
    ax_test.plot([0, 1], [0, 1], 'k--', lw=1, label='Reference')
    ax_test.set_xlabel('1 - Specificity', fontsize=12)
    ax_test.set_ylabel('Sensitivity', fontsize=12)
    ax_test.set_title('ROC Curve - Testing Set', fontsize=14, fontweight='bold')
    ax_test.legend(loc='lower right', fontsize=10)
    ax_test.grid(True, alpha=0.3)
    ax_test.set_xlim([0.0, 1.0])
    ax_test.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_data(FILE_PATH, standardize=True)
    # 打印表格头
    print(f"| Models |  AUC  |  ACC  |  SEN  |  SPE  |F1Score|  PPV  |  NPV  |")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)

    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test)

    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)

    # 绘制ROC曲线
    models = [lr_model, dt_model, rf_model, xgb_model]
    model_names = ['LR', 'DT', 'RF', 'XGBoost']
    plot_roc_curves(models, model_names, X_train, y_train, X_test, y_test)