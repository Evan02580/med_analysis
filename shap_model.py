"""
SHAP 模型解释模块
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 配置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 优先使用黑体，fallback到英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def SHAP_analysis(model, X_train, feature_names=None, random_state=42):
    """使用SHAP进行模型解释

    Args:
        model: 训练好的模型
        X_train: 训练集特征（NumPy数组或DataFrame）
        feature_names: 特征名称列表
        save_path: 保存图片的路径
        random_state: 随机种子
    """
    import shap
    model_name = type(model).__name__
    save_path = f"results/{model_name}_shap_summary.png"

    # 确保中文字体配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 如果是NumPy数组，转换为DataFrame以便显示特征名
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
        if feature_names is None:
            feature_names = X_train_df.columns.tolist()

    # 使用样本进行SHAP分析以提高速度
    # sample_size = min(100, X_train_df.shape[0])
    X_sample = X_train_df ## .sample(n=sample_size, random_state=random_state)

    try:
        # 尝试使用新版本的 SHAP API (0.41+)
        if hasattr(shap, 'Explainer'):
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
        else:
            # 使用旧版本 API
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

        # 处理不同的 SHAP 值格式
        # Random Forest 返回的是 (samples, features, classes) 形状
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 取正类的 SHAP 值 (最后一个类别)
            shap_values = shap_values[:, :, 1]
        # 如果是列表（某些版本的 SHAP），取正类的值
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]

    except Exception as e:
        print(f"SHAP Explainer 初始化失败: {e}")
        print("尝试使用 shap.TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # 处理不同的 SHAP 值格式
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            elif isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as e2:
            print(f"TreeExplainer 也失败了: {e2}")
            print("尝试使用 shap.KernelExplainer...")
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)

            # 对于二分类的 KernelExplainer，取正类的值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

    # 绘制SHAP summary plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # 检查 shap_values 的形状和类型，确保是 2D 数组用于柱状图
        # 如果是 Explanation 对象，先提取 values
        if hasattr(shap_values, 'values'):
            # 新版本 SHAP Explanation 对象
            values = shap_values.values
            # 处理 3D 情况
            if isinstance(values, np.ndarray) and values.ndim == 3:
                values = values[:, :, 1]  # 取正类
            shap.summary_plot(values, X_sample, plot_type="bar", show=False)
        elif isinstance(shap_values, np.ndarray):
            # numpy 数组直接使用
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        else:
            # 其他类型
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)

        # 设置图表标题和字体
        plt.title(f'SHAP Importance {model_name}', fontsize=14, fontweight='bold', pad=20)

        # 获取当前轴并设置中文字体
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontproperties('SimHei')
        for label in ax.get_yticklabels():
            label.set_fontproperties('SimHei')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"绘制SHAP图表失败: {e}")
        print("但SHAP值已成功计算")


def SHAP_waterfall_plot(model, X_train, feature_names=None, sample_idx=0, random_state=42):
    """绘制SHAP Waterfall图（解释单个预测）

    Args:
        model: 训练好的模型
        X_train: 训练集特征（NumPy数组或DataFrame）
        feature_names: 特征名称列表
        sample_idx: 要分析的样本索引
        random_state: 随机种子
    """
    import shap

    # 如果是NumPy数组，转换为DataFrame以便显示特征名
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
        if feature_names is None:
            feature_names = X_train_df.columns.tolist()

    # 使用样本进行SHAP分析以提高速度
    sample_size = min(100, X_train_df.shape[0])
    X_sample = X_train_df.sample(n=sample_size, random_state=random_state)

    try:
        # 计算SHAP值
        model_name = type(model).__name__

        # 尝试创建 TreeExplainer，如果失败则使用 KernelExplainer
        try:
            # 尝试使用 check_additivity 参数（新版本）
            try:
                explainer = shap.TreeExplainer(model, check_additivity=False)
            except TypeError:
                # 旧版本不支持该参数
                explainer = shap.TreeExplainer(model)

            # 计算 SHAP 值，尝试禁用 additivity 检查
            try:
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
            except TypeError:
                # 旧版本不支持 check_additivity 参数
                shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            # TreeExplainer 不支持该模型，使用 KernelExplainer
            print(f"  使用 KernelExplainer for {model_name}...")
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # 处理不同的 SHAP 值格式
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D 数组，取正类 (class 1)
            shap_values_2d = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            # 列表格式，取正类
            shap_values_2d = shap_values[1]
        else:
            # 已经是 2D 数组
            shap_values_2d = shap_values

        # 确保样本索引有效
        if sample_idx >= len(X_sample):
            sample_idx = 0

        # 获取基础值（期望值）
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, np.ndarray):
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value
        else:
            base_value = 0.5

        # 绘制 Waterfall 图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 构建 waterfall 数据
        feature_vals = X_sample.iloc[sample_idx].values
        shap_vals_sample = shap_values_2d[sample_idx]

        # 创建 Explanation 对象并绘制 waterfall
        try:
            # 尝试使用新版本 API
            shap_explanation = shap.Explanation(
                values=shap_vals_sample,
                base_values=base_value,
                data=feature_vals,
                feature_names=feature_names
            )
            shap.plots.waterfall(shap_explanation, show=False)
        except:
            # 备用方案：手动绘制
            # 按 SHAP 值的大小排序特征
            indices = np.argsort(np.abs(shap_vals_sample))[::-1]

            # 计算累积贡献
            values = [base_value]
            labels = ['Base value']
            colors = ['gray']

            cumsum = base_value
            for idx in indices[:15]:  # 只显示前15个特征
                values.append(shap_vals_sample[idx])
                labels.append(f"{feature_names[idx]}\n({feature_vals[idx]:.2f})")
                colors.append('red' if shap_vals_sample[idx] > 0 else 'blue')
                cumsum += shap_vals_sample[idx]

            values.append(cumsum - shap_vals_sample[indices[14:].sum()] if len(indices) > 15 else 0)

            # 绘制条形图
            ax.barh(range(len(values)), values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.axvline(base_value, color='black', linestyle='--', alpha=0.3)
            ax.axvline(cumsum, color='black', linestyle='--', alpha=0.3)

        plt.title(f'SHAP Waterfall Plot - {model_name} (Sample {sample_idx})',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Model Output Value', fontsize=12)
        plt.tight_layout()

        save_path = f"results/{model_name}_shap_waterfall_{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to: {save_path}")
        plt.show()

    except Exception as e:
        print(f"绘制SHAP Waterfall图失败: {e}")

