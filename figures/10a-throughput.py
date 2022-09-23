import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import util
from pylab import text

def set_boxplot_color(bp, color, label):
    plt.plot([], c=color, label=label)

    for attr in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[attr], color='black')
    
    for patch in bp['boxes']:
        patch.set(facecolor=color)  

def main():
    parser = argparse.ArgumentParser(description="Plot throughput comparison")
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()

    util.use_style()

    script_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(script_dir, "compare-throughput-fig10a.csv"))
    figsize = (util.columnwidth_acmart, 1.5)
    dt_urban = data["FLY524_P1_ul"][data["FLY524_P1_ul"].notna()]
    dt_rural = data["FLY526_P1_ul"][data["FLY526_P1_ul"].notna()]
    o2_urban = data["FLY524_P2_ul"][data["FLY524_P2_ul"].notna()]
    o2_rural = data["FLY526_P2_ul"][data["FLY526_P2_ul"].notna()]

    operator1 = [dt_urban, dt_rural]
    operator2 = [o2_urban, o2_rural]
    ticks = ['Urban', 'Rural']

    fig1, ax1 = plt.subplots(figsize=figsize)
    operator1_plot = ax1.boxplot(operator1,
        positions=np.array(np.arange(len(operator1)))*.03-0.0065, 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.009, 
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))

    operator2_plot = ax1.boxplot(operator2,
        positions=np.array(np.arange(len(operator2)))*.03+0.0065, 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.009, 
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))
    
    set_boxplot_color(operator1_plot, util.mpl_colors()[0], 'P1')
    set_boxplot_color(operator2_plot, util.mpl_colors()[1], 'P2')

    text(47, 0.04,'\\textbf{(a)}')
    # set the x label values
    plt.grid(axis='y')
    plt.xlim([-0.75, 50])
    plt.yticks([0, 0.03], ticks, va='center', rotation=90)
    plt.ylim([-0.0175, 0.0495])
    plt.ylabel("")
    plt.xscale("linear")
    util.set_actual_figsize(fig1, figsize)
    
    plt.xlabel("Throughput (Mbps)")
    plt.legend(loc="center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False)
    util.set_actual_figsize(fig1, figsize)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fig1.savefig(args.save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

if __name__ == '__main__':
    main()
