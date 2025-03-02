import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'Times New Roman'

try:
    data = pd.read_excel(r"G:\文件整理\7随机森林结果\全部变量.xlsx", sheet_name="0.6")
except FileNotFoundError:
    print("File path error, please check if the file path is correct.")
    exit()
except KeyError:
    print("The specified sheet does not exist, please check the sheet_name parameter.")
    exit()


subplots_data = [
    {"mat": data['NF_RAT'].values, "map": data['NF_RSSR'].values, "sensitivity_change": data['NF_RT'].values,
     "title": "NF_RAT vs NF_RSSR", "xlabel": "NF_RSSR", "ylabel": "NF_RAT", "cmap": "Oranges"},
    {"mat": data['NF_RAT'].values, "map": data['NF_RVPD'].values, "sensitivity_change": data['NF_RT'].values,
     "title": "NF_RAT vs NF_RVPD", "xlabel": "NF_RVPD", "ylabel": "NF_RAT", "cmap": "Oranges"},
    {"mat": data['PF_RAT'].values, "map": data['PF_RSSR'].values, "sensitivity_change": data['PF_RT'].values,
     "title": "PF_RAT vs PF_RSSR", "xlabel": "PF_RSSR", "ylabel": "PF_RAT", "cmap": "Blues"},
    {"mat": data['PF_RAT'].values, "map": data['PF_RVPD'].values, "sensitivity_change": data['PF_RT'].values,
     "title": "PF_RAT vs PF_RVPD", "xlabel": "PF_RVPD", "ylabel": "PF_RAT", "cmap": "Blues"}
]


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

x_min1 = min(min(data['NF_RSSR'].values), min(data['PF_RSSR'].values))
x_max1 = max(max(data['NF_RSSR'].values), max(data['PF_RSSR'].values))

x_min2 = min(data['PF_RVPD'].values)
x_max2 = max(data['PF_RVPD'].values)

vmin = 0
vmax = 18

for idx, subplot in enumerate(subplots_data):
    mat = subplot["mat"]
    map = subplot["map"]
    sensitivity_change = subplot["sensitivity_change"]
    cmap = subplot["cmap"]

    # Check if data is empty
    if len(mat) == 0 or len(map) == 0 or len(sensitivity_change) == 0:
        print(f"Subplot {subplot['title']} data is empty, skipping plot.")
        continue

    scatter = axes[idx].scatter(map, mat, c=sensitivity_change, cmap=cmap, vmin=vmin, vmax=vmax, s=30, edgecolors='none', alpha=0)

    axes[idx].set_xlabel(subplot["xlabel"], fontsize=14, family='Times New Roman')
    axes[idx].set_ylabel(subplot["ylabel"], fontsize=14, family='Times New Roman')
    axes[idx].tick_params(axis='both', labelsize=15)

    if idx == 0 or idx == 2:
        axes[idx].set_xlim(x_min1, x_max1)
    elif idx == 1 or idx == 3:
        axes[idx].set_xlim(x_min2, x_max2)

    num_bins_x = 24
    num_bins_y = 24

    x_bins = np.linspace(axes[idx].get_xlim()[0], axes[idx].get_xlim()[1], num_bins_x + 1)
    y_bins = np.linspace(axes[idx].get_ylim()[0], axes[idx].get_ylim()[1], num_bins_y + 1)

    means = np.zeros((num_bins_y, num_bins_x))

    for i in range(num_bins_x):
        for j in range(num_bins_y):

            x_range = (x_bins[i], x_bins[i + 1])
            y_range = (y_bins[j], y_bins[j + 1])

            mask = (map >= x_range[0]) & (map < x_range[1]) & (mat >= y_range[0]) & (mat < y_range[1])

            if np.any(mask):
                means[j, i] = np.mean(sensitivity_change[mask])
            else:
                means[j, i] = np.nan

    cax = axes[idx].imshow(means, origin="lower", aspect="auto", cmap=cmap, extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]))
    cbar = fig.colorbar(cax, ax=axes[idx], orientation="vertical", pad=0.15)
    cbar.set_label('RT', fontsize=14, family='Times New Roman')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(np.arange(vmin, vmax + 1, 2))
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()


    overall_mean = np.nanmean(means)
    overall_std = np.nanstd(means)
    overall_cv = overall_std / overall_mean if overall_mean != 0 else 0
    print(f"Overall CV for subplot {subplot['title']}: {overall_cv:.2f}")


    ax_top = axes[idx].inset_axes([0, 1.05, 1, 0.1])
    ax_top.scatter(np.arange(1, 25), np.nanmean(means, axis=0), color='black', s=2)
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax_top.set_xticks(np.arange(1, 25, 1))
    ax_top.tick_params(axis='y', labelsize=16)
    ax_top.set_xlim(0.5, 24.5)
    ax_top.set_xticklabels([])

    row_means = np.nanmean(means, axis=1)
    ax_side = axes[idx].inset_axes([1.05, 0, 0.1, 1])
    ax_side.scatter(row_means, np.arange(1, 25), color='black', s=2)
    ax_side.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax_side.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax_side.set_yticks(np.arange(1, 25, 1))
    ax_side.set_ylim(0.5, 24.5)
    ax_side.tick_params(axis='x', labelsize=16)
    ax_side.set_yticklabels([])

    if idx ==2 or idx ==3:
        axes[idx].text(0.13, 0.90, f'|PF_CV| = {overall_cv:.2f}', transform=axes[idx].transAxes, fontsize=16, fontweight='bold',
                   color='black', family='Times New Roman')
    else:
        axes[idx].text(0.13, 0.90, f'|NF_CV| = {overall_cv:.2f}', transform=axes[idx].transAxes, fontsize=16,
                       fontweight='bold',
                       color='black', family='Times New Roman')
    labels = ['a', 'b', 'c', 'd']
    axes[idx].text(0.05, 0.90, f'{labels[idx]}', transform=axes[idx].transAxes, fontsize=20, fontweight='bold',
                   color='black', family='Times New Roman')


plt.tight_layout()
plt.show()
