import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


df = pd.read_csv('marketing_campaign_CLEANED.csv', sep=',')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
ref_date = pd.to_datetime('2021-01-01')
df['DaysSinceSignup'] = (ref_date - df['Dt_Customer']).dt.days
df['Age'] = 2021 - df['Year_Birth']
df = df[df['Age'] <= 100]
df = df.dropna(subset=['Income'])
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)
df['TotalSpent'] = df[[
    'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds'
]].sum(axis=1)
df['HighSpenderFlag'] = (df['TotalSpent'] > 250).astype(int)

features = [
    'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'DaysSinceSignup',
    'NumWebPurchases', 'NumStorePurchases', 'NumDealsPurchases',
    'TotalSpent', 'HighSpenderFlag'
] + [col for col in df.columns if col.startswith('Education_') or col.startswith('Marital_Status_')]
target = 'MntWines'
Setting_Amount = 30
df['TotalSpendingAll'] = df[[
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]].sum(axis=1)

df_active = df[df['TotalSpendingAll'] >= Setting_Amount].copy()
X = df_active[features]
y = df_active[target]
print("Features in X:", X.columns.tolist())      # 一眼查看是否含 'ID' 或 'Dt_Customer'
rf_init = RandomForestRegressor(n_estimators=100, random_state=42)
rf_init.fit(X, y)
importances = rf_init.feature_importances_
print("Feature Importances (Initial):", importances)
features_all = X.columns
coef_table = pd.DataFrame({
    "Feature": features_all,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
important_features = coef_table[coef_table['Importance'] > 0.01]['Feature'].tolist()
X_reduced = X[important_features]
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel R²: {r2:.3f}, MSE: {mse:.2f}")
coef_reduced = pd.DataFrame({
    "Feature": X_reduced.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance Table")
print(coef_reduced)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual MntWines")
plt.ylabel("Predicted MntWines")
plt.title("Predicted vs Actual Red Wine Spending (Reduced Features)")
plt.grid(True)
plt.tight_layout()
plt.show()
cv_scores = cross_val_score(rf_model, X_reduced, y, cv=5, scoring='r2')
print("Cross_Validation R² score:", cv_scores)
print(f"mean R²:{cv_scores.mean():.3f}")
import joblib
joblib.dump(rf_model, 'RFMODEL.pkl')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)
param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(
    rf, param_grid,
    cv=5,               
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Parameter:", grid.best_params_)
print(f"Validator Set R²: {grid.best_score_:.3f}")
y_pred = best_model.predict(X_test)
print(f"Test Set R²: {r2_score(y_test, y_pred):.3f}")
print(f"Test Set MSE: {mean_squared_error(y_test, y_pred):.2f}")
import matplotlib.pyplot as plt
import seaborn as sns
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=X_reduced.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(6,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="Blues_d")
plt.xlabel("Feature Importance")
plt.title("Top-10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
import numpy as np
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residual (Actual − Predicted)")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()



