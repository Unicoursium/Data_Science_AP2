import pandas as pd

# 读取CSV文件
df = pd.read_csv("marketing_campaign_NEW.csv", sep='\t')

# 删除 Income 为缺失值（NaN）的行
df_cleaned = df.dropna(subset=['Income'])

# 可选：保存为新的CSV文件
df_cleaned.to_csv("marketing_campaign_CLEANED.csv", index=False)
print(f"原始行数: {len(df)}, 删除后行数: {len(df_cleaned)}")
