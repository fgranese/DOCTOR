import matplotlib.pyplot as plt
import numpy as np
from utils.files_utils import get_plot_path
from pathlib import Path
import os

def plot_font_setup(ax_plt, legend_title, x_label, y_label, tick_size=15, legend_title_size=65, legend_content_size=65, label_size=18):
    ax_plt.tick_params(axis='both', which='both', labelsize=tick_size)
    h, l = ax_plt.get_legend_handles_labels()
    ax_plt.legend(h, l, title=legend_title, prop={'size': legend_content_size}, title_fontsize=legend_title_size)
    plt.ylabel(y_label, fontsize=label_size)
    plt.xlabel(x_label, fontsize=label_size)

def plot_rocs(dataset_name, T_0, T_1, labels, colors, plot_type):

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    plot_font_setup(ax_plt=ax1, legend_title='', x_label='', y_label='')

    for i in range(len(T_0)):
        ################ To plot plot zoomed ############
        X = []
        Y = []
        for j in range(len(T_1[i])):
            if T_1[i][j] > 0.8:
                X.append(T_0[i][j])
                Y.append(T_1[i][j])
        ax1.plot(X, Y, label=labels[i], color=colors[i])
        ##################################################
        # De-comment this line and comment the previous to plot without zoom
        #ax1.plot(T_0[i], T_1[i], label=labels[i], color=colors[i])

        x = np.interp(0.95, np.sort(T_1[i]), np.sort(T_0[i]))
        ax1.scatter(x=x, y=0.95, color=colors[i], marker='x')

    for ax in [ax1]:
        ax.set(xlabel='FRR', ylabel='TRR')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(title='', prop={'size': 18}, title_fontsize=18)

    img_path = get_plot_path(dataset_name=dataset_name, plot_type=plot_type)
    plt.subplots_adjust(bottom=0.15, wspace=0.5)
    plt.grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.2)

    plt.axhline(y=0.95, color='red', linestyle=':')

    p = Path(img_path)
    if not os.path.exists(p.parent):
        os.makedirs(p.parent)
    plt.savefig(img_path)

    plt.show()
