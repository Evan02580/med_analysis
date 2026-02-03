import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def smart_impute(file_path, output_path, keep_cols, n_neighbors=5, noise_level=0.05):
    # 1. 读取数据
    df = pd.read_csv(file_path)

    # --- 列名映射与筛选 ---
    # 注意：CSV中的列名可能与你提供的KEEP_COLS略有不同（例如 "length" vs "length mm"）
    # 这里做一个交集处理，只保留CSV里确实存在的列，防止报错
    valid_cols = [c for c in keep_cols if c in df.columns]
    missing_cols = set(keep_cols) - set(df.columns)
    if missing_cols:
        print(f"⚠️ 提示: 以下列在CSV中未找到，将被跳过: {missing_cols}")

    df = df[valid_cols].copy()

    # 记录原始信息的字典，用于最后还原限制
    col_limits = {}
    col_dtypes = {}  # 记录原本是否为整数

    for col in df.columns:
        # 跳过非数值列的极值计算（稍后编码后再算）
        if pd.api.types.is_numeric_dtype(df[col]):
            col_limits[col] = {'min': df[col].min(), 'max': df[col].max()}
            # 判断是否应该强制取整：如果该列现有数据全是整数（忽略NaN），则标记为整数列
            valid_values = df[col].dropna()
            is_int = np.all(valid_values % 1 == 0)
            col_dtypes[col] = 'int' if is_int else 'float'

    # --- 预处理：处理ID和文本 ---
    # 假设 'Number' 是ID列，不参与KNN距离计算，先分离
    id_col = None
    if 'Number' in df.columns:
        id_col = df['Number']
        df_calc = df.drop(columns=['Number'])
    else:
        df_calc = df.copy()

    # 处理文本列 (Label Encoding)
    object_cols = df_calc.select_dtypes(include=['object']).columns
    encoders = {}
    for col in object_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        # 转字符串处理，防止混合类型报错
        df_calc[col] = enc.fit_transform(df_calc[[col]].astype(str))
        encoders[col] = enc
        # 记录编码后的极大极小值，用于限制范围
        col_limits[col] = {'min': df_calc[col].min(), 'max': df_calc[col].max()}
        col_dtypes[col] = 'int'  # 文本分类编码后肯定是整数

    # 归一化 (MinMax)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_calc), columns=df_calc.columns)
    missing_mask = df_calc.isna()  # 记录哪里是原本缺失的

    # --- 核心：KNN 填补 ---
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled_array = imputer.fit_transform(df_scaled)
    df_filled = pd.DataFrame(filled_array, columns=df_calc.columns)

    # --- 核心：加入噪声并还原 ---
    # 反归一化回到原始尺度
    df_final = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df_calc.columns)

    for col in df_final.columns:
        # 1. 获取原本缺失的行
        missing_rows = missing_mask[col]
        if not missing_rows.any():
            continue

        # 2. 只有缺失位置加噪声 (Noise Injection)
        std = df_final[col].std()
        noise = np.random.normal(0, std * noise_level, size=missing_rows.sum())
        df_final.loc[missing_rows, col] += noise

        # 3. 约束限制 (Clipping) - 确保不超出原始数据的 [min, max]
        if col in col_limits:
            lower = col_limits[col]['min']
            upper = col_limits[col]['max']
            df_final.loc[missing_rows, col] = df_final.loc[missing_rows, col].clip(lower, upper)

        # 4. 化整 (Rounding) - 如果原本是整数列（如性别、评分），则四舍五入取整
        if col in col_dtypes and col_dtypes[col] == 'int':
            df_final.loc[missing_rows, col] = df_final.loc[missing_rows, col].round()

    # --- 后处理：解码文本与合并ID ---
    # 将编码后的数字还原回文本 (如 0 -> 'A3')
    for col in object_cols:
        # 先取整确保是合法的索引
        df_final[col] = df_final[col].round().astype(int)
        # 还原
        try:
            df_final[col] = encoders[col].inverse_transform(df_final[[col]])
        except:
            pass

            # 拼回ID列
    if id_col is not None:
        df_final.insert(0, 'Number', id_col)

    # 取整：age	sex	length	height	weight	smoke	drink	hypertension	diabetes	time	胫骨前肌	拇背伸肌肌力	疼痛	感觉障碍	椎间盘退化	segment	是否脱出/游离
    int_cols = ["age", "sex", "height", "weight", "smoke", "drink",
                "hypertension", "diabetes", "time", "胫骨前肌", "拇背伸肌肌力", "疼痛", "疼痛", "感觉障碍",
                "椎间盘退化", "segment", "是否脱出/游离", "JOA", "NRS/VAS"]
    for col in int_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].round().astype(int)

    for col in ["椎管骨性直径", "椎管最小直径", "椎管面积"]:
        if col in df_final.columns:
            df_final[col] = df_final[col].round(2)

    df_final['length'] = df_final['length'].round(2)
    df_final['BMI'] = df_final['weight'] / ((df_final['height'] / 100) ** 2)
    df_final['BMI'] = df_final['BMI'].round(2)
    # 大于90为1，否则为0
    df_final['time'] = df_final['time'].apply(lambda x: 1 if x > 90 else 0)

    # 保存
    df_final.to_csv(output_path, index=False)
    print(f"✅ 处理完成！已保存至: {output_path}")
    return df_final


# --- 配置参数 ---
# 注意：根据你的CSV，我微调了列表以匹配可能的实际列名（例如去掉问号），
# 如果你的CSV里确实带问号，请改回原样。
KEEP_COLS = [
    "Number", "name", "age", "sex", "length", "dropped", "height", "weight", "smoke", "drink", "hypertension",
    "diabetes", "time", "胫骨前肌", "拇背伸肌肌力", "疼痛", "感觉障碍", "椎间盘退化", "segment", "是否脱出/游离", "狭窄分级",
    "MSU分级", "MSU分区", "椎管骨性直径", "椎管最小直径", "椎管面积", "JOA", "NRS/VAS"
]

# 运行
# noise_level=0.05 代表允许数据在标准差的5%范围内波动
# n_neighbors=5 寻找最近的5个相似样本
df_result = smart_impute('datasets/raw_data.csv', 'datasets/filled_data_v3.csv', KEEP_COLS, n_neighbors=5, noise_level=0.05)

# 打印前几行查看效果
print(df_result.head())