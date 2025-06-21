import joblib
import pandas as pd

# 加载模型
model = joblib.load('RFMODEL.pkl')  # 请确保模型文件名和路径一致

# 创建带列名的输入特征（顺序要和训练时一样）
# 替换下面的值为你自己的数据
input_data = pd.DataFrame([[
    50000,      # Income
    8,         # NumStorePurchases
    1365,        # DaysSinceSignup
    1,       # NumWebPurchases
    500,          # TotalSpent
    20,         # Age
    3,         # Recency
    0,          # Education_PhD（是否为PhD，1为是，0为否）
    10           # NumDealsPurchases
]], columns=[
    'Income', 'NumStorePurchases', 'DaysSinceSignup',
    'NumWebPurchases','TotalSpent', 'Age',
    'Recency', 'Education_PhD', 'NumDealsPurchases'
])

# 进行预测
prediction = model.predict(input_data)

print("预测结果：", prediction)
