import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# 1. Read the data
df = pd.read_csv('datasets/filled_data_v3 0108.csv', encoding='utf-8-sig')

# 2. Filter samples where dropped=0 and randomly sample 60 rows
dropped_0_samples = df[df['dropped'] == 0].copy()
sampled_60 = dropped_0_samples.sample(n=60, random_state=42).copy()

# 3. Create new Numbers: 0001~0060
sampled_60['Number'] = [f'{i:04d}' for i in range(1, 61)]

# 4. Identify column types
# Columns that need rounding to 2 decimal places
decimal_2_cols = ['length', '椎管骨性直径', '椎管最小直径', '椎管面积']

# Categorical and binary columns (no noise)
categorical_cols = ['dropped', 'sex', 'smoke', 'drink', 'hypertension', 'diabetes', 'time',
                   '胫骨前肌', '拇背伸肌肌力', '疼痛', '感觉障碍', '椎间盘退化', 'segment',
                   '是否脱出/游离', '狭窄分级', 'MSU分级', 'MSU分区', 'JOA', 'NRS/VAS']

# Integer columns (will add noise and round)
integer_cols = ['age', 'height', 'weight']

# All numeric columns except Number, dropped, and BMI
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove columns that shouldn't get noise
noise_cols = [col for col in all_numeric_cols
              if col not in ['Number', 'dropped', 'BMI'] and col not in categorical_cols]

# 5. Add random noise to appropriate columns
for col in noise_cols:
    if col in sampled_60.columns:
        original_values = sampled_60[col].values

        # Get min and max for this column from entire dataset
        col_min = df[col].min()
        col_max = df[col].max()
        col_range = col_max - col_min

        # Add noise: ±5% of the column range
        noise_scale = col_range * 0.05
        noise = np.random.normal(0, noise_scale, size=len(original_values))
        noisy_values = original_values + noise

        # Clip to ensure values stay within original range
        noisy_values = np.clip(noisy_values, col_min, col_max)

        # Round based on column type
        if col in decimal_2_cols:
            noisy_values = np.round(noisy_values, 2)
        elif col in integer_cols:
            noisy_values = np.round(noisy_values).astype(int)
        else:
            # For categorical-like numeric columns, round to nearest integer
            noisy_values = np.round(noisy_values).astype(int)

        sampled_60[col] = noisy_values

# 6. Recalculate BMI based on height and weight
# BMI = weight / (height/100)^2
sampled_60['BMI'] = sampled_60['weight'] / ((sampled_60['height'] / 100) ** 2)
sampled_60['BMI'] = sampled_60['BMI'].round(2)

# 7. Ensure proper column order (Number first)
column_order = ['Number'] + [col for col in sampled_60.columns if col != 'Number']
sampled_60 = sampled_60[column_order]

# 8. Save to CSV
sampled_60.to_csv('datasets/oversample_60.csv', index=False, encoding='utf-8-sig')

print(f"Successfully created oversample_60.csv with {len(sampled_60)} samples")
print(f"\nSample preview:")
print(sampled_60[['Number', 'age', 'length', '椎管骨性直径', 'height', 'weight', 'BMI', 'dropped']].head(10))
print(f"\nColumn types after processing:")
print(f"Integer columns: {integer_cols}")
print(f"Decimal (2 places) columns: {decimal_2_cols}")
print(f"\nVerifying Number format:")
print(f"First 3 Numbers: {sampled_60['Number'].head(3).tolist()}")
print(f"Last 3 Numbers: {sampled_60['Number'].tail(3).tolist()}")


