import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.utils import resample

plt.rc('font', family='Times New Roman')
def remove_outliers(df, columns):
    # 筛选指定列的异常值
    for column in columns:
        mean_val = df[column].mean()
        std_val = df[column].std()
        df = df[(df[column] >= mean_val - 3 * std_val) & (df[column] <= mean_val + 3 * std_val)]
    return df
def smooth_curve(x, y, points=800, window_size=20):
    x_new = np.linspace(x.min(), x.max(), points)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)

    # 使用移动平均平滑曲线
    y_smooth = np.convolve(y_smooth, np.ones(window_size) / window_size, mode='same')

    return x_new, y_smooth

data_path = r"G:\文件整理\7随机森林结果\非气候变量与残差.xlsx"
data = pd.read_excel(data_path)
x_columns = ['PF_ROOT', 'NF_ROOT', 'PF_TD', 'NF_TD',
             'PF_GPP', 'NF_GPP','PF_TA', 'NF_TA', 'PF_MS', 'NF_MS',
             'PF_AGB', 'NF_AGB', 'PF_TH', 'NF_TH',  'PF_LAI', 'NF_LAI', 'PF_WUE', 'NF_WUE']
y_columns = ['PF_Residuals', 'NF_Residuals']
data_filtered_residuals = remove_outliers(data, y_columns)
data_filtered_all = remove_outliers(data_filtered_residuals, x_columns + y_columns)
models = {}
for y_col in y_columns:
    X = data_filtered_all[x_columns]
    y = data_filtered_all[y_col]

    rf_model = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42)
    rf_model.fit(X, y)
    models[y_col] = rf_model

def compute_partial_dependence_with_ci(model, X, feature, n_bootstrap=100, ci=0.95):

    pd_results = partial_dependence(model, X=X, features=[feature])
    x_vals = pd_results['values'][0]
    y_vals = pd_results['average'][0]

    y_samples = []
    for _ in range(n_bootstrap):
        X_resampled = resample(X)
        pd_results_bootstrap = partial_dependence(model, X=X_resampled, features=[feature])
        y_samples.append(pd_results_bootstrap['average'][0])

    y_samples = np.array(y_samples)
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    y_lower = np.percentile(y_samples, lower_percentile, axis=0)
    y_upper = np.percentile(y_samples, upper_percentile, axis=0)

    return x_vals, y_vals, y_lower, y_upper

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

y_all = []
for i, (pf_feature, nf_feature) in enumerate(zip(x_columns[::2], x_columns[1::2])):
    x_pf, y_pf, y_pf_lower, y_pf_upper = compute_partial_dependence_with_ci(
        models['PF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=pf_feature
    )
    x_nf, y_nf, y_nf_lower, y_nf_upper = compute_partial_dependence_with_ci(
        models['NF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=nf_feature
    )
    y_all.append((y_pf, y_nf))


y_min = min([min(y_pf.min(), y_nf.min()) for y_pf, y_nf in y_all])
y_max = max([max(y_pf.max(), y_nf.max()) for y_pf, y_nf in y_all])


yticks = np.around(np.linspace(y_min, y_max, 5), decimals=2)


annotations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
for i, (pf_feature, nf_feature) in enumerate(zip(x_columns[::2], x_columns[1::2])):
    ax = axes[i]
    x_pf, y_pf, y_pf_lower, y_pf_upper = compute_partial_dependence_with_ci(
        models['PF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=pf_feature
    )
    x_pf_smooth, y_pf_smooth = smooth_curve(x_pf, y_pf)
    _, y_pf_lower_smooth = smooth_curve(x_pf, y_pf_lower)
    _, y_pf_upper_smooth = smooth_curve(x_pf, y_pf_upper)
    ax.plot(x_pf_smooth, y_pf_smooth, color=(109 / 255, 175 / 255, 215 / 255), label=f"{pf_feature}")
    ax.fill_between(x_pf_smooth, y_pf_lower_smooth, y_pf_upper_smooth, color=(109 / 255, 175 / 255, 215 / 255),
                    alpha=0.2)
    x_nf, y_nf, y_nf_lower, y_nf_upper = compute_partial_dependence_with_ci(
        models['NF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=nf_feature
    )
    x_nf_smooth, y_nf_smooth = smooth_curve(x_nf, y_nf)
    _, y_nf_lower_smooth = smooth_curve(x_nf, y_nf_lower)
    _, y_nf_upper_smooth = smooth_curve(x_nf, y_nf_upper)
    ax.plot(x_nf_smooth, y_nf_smooth, color=(253 / 255, 181 / 255, 118 / 255), label=f"{nf_feature}")
    ax.fill_between(x_nf_smooth, y_nf_lower_smooth, y_nf_upper_smooth, color=(253 / 255, 181 / 255, 118 / 255),
                    alpha=0.2)
    ax.set_yticks(yticks)
    ax.text(0.02, 0.98, annotations[i], transform=ax.transAxes,
            fontsize=33, fontweight='bold', va='top', ha='left')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

# 调整布局
plt.tight_layout()
plt.savefig(r"G:\文件整理\7随机森林结果\补充实验\图\4_subplots_with_CI.png", dpi=300, bbox_inches="tight")
plt.show()
