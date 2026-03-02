"""
SHAP 模型解释模块
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def SHAP_analysis(model, X_train, feature_names=None, sample_idx=0, random_state=42):
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



def SHAP_waterfall_plot(model, X_train, y_train, feature_names=None, sample_idx=0, random_state=42):
    """绘制SHAP Waterfall图（解释单个预测）

    Args:
        model: 训练好的模型
        X_train: 训练集特征（NumPy数组或DataFrame）
        y_train: 训练集标签（用于显示真实标签）
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
    #sample_size = min(100, X_train_df.shape[0])
    X_sample = X_train_df  #.sample(n=sample_size, random_state=random_state)

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
        # ====== 取该 sample 的真实标签 y_true ======
        y_true = y_train[sample_idx]

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

        plt.title(f'Waterfall - {model_name} (Sample {sample_idx})',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(f'True Value: {y_true}', fontsize=12)
        plt.tight_layout()

        save_path = f"results/waterfall_new/{model_name}_shap_waterfall_{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to: {save_path}")
        plt.show()


    except Exception as e:
        print(f"绘制SHAP Waterfall图失败: {e}")


def SHAP_both(
    model,
    X_train,
    feature_names=None,
    random_state=42,
    sample_idx=0,            # ✅ 可选样本（默认0）
    X_train_raw=None,        # ✅ 未标准化的原始特征（用于展示）
    y_true=None,             # ✅ 真实y（用于标注）
    max_display=10,          # ✅ waterfall显示前多少个特征
    if_sum=True             # ✅ 是否绘制 summary（默认True）
):
    import os
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    model_name = type(model).__name__
    os.makedirs("results", exist_ok=True)

    summary_path = f"results/summary/{model_name}_shap_summary.png"
    waterfall_path = f"results/waterfall/{model_name}_shap_waterfall_sample{sample_idx}.png"

    # 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 标准化后的 X -> DataFrame（用于SHAP）
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_scaled_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_scaled_df = X_train.copy()
        if feature_names is None:
            feature_names = X_scaled_df.columns.tolist()

    # raw X -> DataFrame（用于展示原始值）
    if X_train_raw is not None:
        if isinstance(X_train_raw, np.ndarray):
            X_raw_df = pd.DataFrame(X_train_raw, columns=feature_names)
        else:
            X_raw_df = X_train_raw.copy()
            # 尽量对齐列顺序
            X_raw_df = X_raw_df[feature_names]
    else:
        X_raw_df = None

    # safety：sample_idx 越界时兜底
    if sample_idx < 0 or sample_idx >= len(X_scaled_df):
        sample_idx = 0

    # =========================
    # 1) 计算 SHAP
    # =========================
    X_sample = X_scaled_df  # 你原先是全量训练集；如果想加速可在外部做sample
    explainer = None
    shap_values = None

    try:
        # 优先新API
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)  # Explanation
    except Exception:
        # 兜底：TreeExplainer / KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
            base = explainer.expected_value
            # 转成 Explanation，后面 waterfall 更统一
            shap_values = shap.Explanation(
                values=sv,
                base_values=base,
                data=X_sample.values,
                feature_names=feature_names
            )
        except Exception:
            # 最后兜底（慢）：KernelExplainer
            f = (lambda x: model.predict_proba(x)[:, 1]) if hasattr(model, "predict_proba") else model.predict
            background = shap.sample(X_sample, min(100, len(X_sample)), random_state=random_state)
            explainer = shap.KernelExplainer(f, background)
            sv = explainer.shap_values(X_sample)
            base = explainer.expected_value
            shap_values = shap.Explanation(
                values=sv,
                base_values=base,
                data=X_sample.values,
                feature_names=feature_names
            )

    # =========================
    # 2) 统一取“正类”输出（如果是多输出/多类别）
    # =========================
    def to_positive_class(expl):
        # expl: Explanation
        vals = expl.values
        base = expl.base_values

        # 多分类/二分类prob输出： (n, p, k)
        if isinstance(vals, np.ndarray) and vals.ndim == 3:
            k = vals.shape[2]
            pos_k = 1 if k > 1 else 0
            vals2 = vals[:, :, pos_k]
            # base 可能是 (n,k) 或 (k,)
            if isinstance(base, np.ndarray) and base.ndim == 2:
                base2 = base[:, pos_k]
            elif isinstance(base, np.ndarray) and base.ndim == 1 and base.shape[0] == k:
                base2 = base[pos_k]
            else:
                base2 = base
            return shap.Explanation(
                values=vals2,
                base_values=base2,
                data=expl.data,
                feature_names=expl.feature_names,
                display_data=getattr(expl, "display_data", None)
            )

        # list（旧API常见）：[class0, class1]
        if isinstance(vals, list):
            pos = 1 if len(vals) > 1 else 0
            base2 = base[pos] if isinstance(base, (list, np.ndarray)) and len(np.atleast_1d(base)) > 1 else base
            return shap.Explanation(
                values=np.array(vals[pos]),
                base_values=base2,
                data=expl.data,
                feature_names=expl.feature_names,
                display_data=getattr(expl, "display_data", None)
            )

        return expl

    shap_values_pos = to_positive_class(shap_values)

    # =========================
    # 3) 画 summary（bar）
    # =========================
    if if_sum:
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_pos.values, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Importance {model_name}", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {summary_path}")
            plt.show()
        except Exception as e:
            print(f"绘制SHAP summary失败: {e}")

    # =========================
    # 4) 画 waterfall（单样本） + 原始值 + y/ŷ标注
    # =========================
    try:
        sv_one = shap_values_pos[sample_idx]

        # 注入 display_data（用于展示 raw 值，不影响 SHAP 加和）
        if X_raw_df is not None:
            raw_row = X_raw_df.iloc[sample_idx].values
            sv_one = shap.Explanation(
                values=sv_one.values,
                base_values=sv_one.base_values,
                data=sv_one.data,
                display_data=raw_row,              # ✅ 这里会显示“原始值”
                feature_names=feature_names
            )

        # y真实值 / 预测值
        y_t = None
        if y_true is not None:
            y_t = int(y_true.iloc[sample_idx]) if hasattr(y_true, "iloc") else int(y_true[sample_idx])

        x_row = X_scaled_df.iloc[[sample_idx]].values
        y_pred = int(model.predict(x_row)[0]) if hasattr(model, "predict") else None
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = float(model.predict_proba(x_row)[0, 1])

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv_one, max_display=max_display, show=False)

        # 在标题处标注 y 与预测
        extra = []
        if y_t is not None:
            extra.append(f"y_true={y_t}")
        if y_pred is not None:
            extra.append(f"y_pred={y_pred}")
        if y_proba is not None:
            extra.append(f"p(y=1)={y_proba:.3f}")
        extra_txt = "  |  ".join(extra)

        plt.title(f"SHAP Waterfall {model_name}  (sample={sample_idx})\n{extra_txt}", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to: {waterfall_path}")
        plt.show()

    except Exception as e:
        print(f"绘制SHAP waterfall失败: {e}")