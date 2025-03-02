import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def remove_outliers(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    return df[(df[column] >= mean_val - 3 * std_val) & (df[column] <= mean_val + 3 * std_val)]

def train_and_evaluate_model(X, y, model):
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    return r2, mse, y_pred

file_path = r"G:\文件整理\7随机森林结果\全部变量.xlsx"
data_06 = pd.read_excel(file_path)

filtered_data_06 = remove_outliers(data_06, 'PF_RT')

climate_variables = ['PF_RAT', 'PF_RVPD', 'PF_RSSR', 'PF_RPRE', 'PF_SF', 'PF_DI', 'PF_DF']
non_climate_variables = ['PF_AGB', 'PF_GPP', 'PF_TA', 'PF_TH', 'PF_LAI', 'PF_TD', 'PF_ROOT', 'PF_WUE', 'PF_MS']

X_climate_06 = filtered_data_06[climate_variables]
y_06 = filtered_data_06['PF_RT']

rf_model_06 = RandomForestRegressor(n_estimators=100, random_state=42)

r2_climate_06, mse_climate_06, y_pred_climate_06 = train_and_evaluate_model(X_climate_06, y_06, rf_model_06)
print('第一次拟合（气候变量）：')
print("随机森林模型（气候变量）的R^2：", r2_climate_06)
print("随机森林模型（气候变量）的均方误差 (MSE)：", mse_climate_06)
residuals_06 = y_06 - y_pred_climate_06
X_non_climate_06 = filtered_data_06[non_climate_variables]
rf_model_nc_06 = RandomForestRegressor(n_estimators=100, random_state=42)
r2_non_climate_06, mse_non_climate_06, y_pred_non_climate_06 = train_and_evaluate_model(X_non_climate_06, residuals_06, rf_model_nc_06)
print('第二次拟合（0.6 表非气候变量）：')
print("随机森林模型（非气候变量对残差）的R^2：", r2_non_climate_06)
print("随机森林模型（非气候变量对残差）的均方误差 (MSE)：", mse_non_climate_06)
output_file = r"G:\文件整理\7随机森林结果\PF非气候变量与残差.xlsx"
combined_df_06 = X_non_climate_06.copy()
combined_df_06['Residuals'] = residuals_06.values
with pd.ExcelWriter(output_file) as writer:
    combined_df_06.to_excel(writer, sheet_name="非气候变量与残差", index=False)

print(f"非气候变量和残差值已保存到同一个工作表：{output_file}")

