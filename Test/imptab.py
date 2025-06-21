import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# ---------- 1. 读取 & 预处理 ----------
df = pd.read_csv('marketing_campaign_CLEANED.csv', sep=',')

# 日期 → 天数
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['DaysSinceSignup'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days

# 年龄、异常值、缺失
df['Age'] = 2025 - df['Year_Birth']
df = df[df['Age'] <= 100]
df = df.dropna(subset=['Income'])

# One-Hot
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

# TotalSpent 与活跃过滤
df['TotalSpent'] = df[['MntFruits','MntMeatProducts','MntFishProducts',
                       'MntSweetProducts','MntGoldProds']].sum(axis=1)
df['TotalSpendingAll'] = df[['MntWines','MntFruits','MntMeatProducts',
                             'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
df = df[df['TotalSpendingAll'] >= 50].copy()

# 删除原始日期列
df = df.drop(columns=['Dt_Customer'])

# ---------- 2. 特征 / 目标 ----------
y = df['MntWines'].astype(float)
X = df.drop(['MntWines', 'TotalSpendingAll'], axis=1)

# ---------- 3. 随机森林计算重要性 ----------
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X, y)

imp_table = (
    pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    })
    .sort_values(by='Importance', ascending=False)
    .reset_index(drop=True)
)

# ---------- 4. 打印 & 可保存 ----------
print(imp_table.head(25))         # 查看前 25 特征
# imp_table.to_csv('feature_importance.csv', index=False)  # 如需存文件
