import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump, load

data = pd.read_excel(r"F:\计算代码\随机森林\全部变量.xlsx")

X = data[['RAT', 'RVPD', 'RSSR']]
y = data["NF_PF"]

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

param_grid2 = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2']
}

model = RandomForestRegressor()

grid_search = GridSearchCV(model, param_grid2, cv=5, n_jobs=-1)

grid_search.fit(X, y)

print("最佳参数组合:", grid_search.best_params_)

score = grid_search.score(X, y)
print("模型得分:", score)

feature_importances = grid_search.best_estimator_.feature_importances_
for i, feature_name in enumerate(X.columns):
    print("特征:", feature_name, "重要性:", feature_importances[i])

