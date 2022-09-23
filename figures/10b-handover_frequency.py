import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import util
from pylab import text
pd.set_option('display.max_rows', None)

def custom_round(x, base=20):
    return int(base * round(float(x)/base))

def set_boxplot_color(bp, color, label):
    plt.plot([], c=color, label=label)

    for attr in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[attr], color='black')
    
    for patch in bp['boxes']:
        patch.set(facecolor=color)  

def main():
    parser = argparse.ArgumentParser(description="Plot the handover frequency")
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()

    corr_param = "relative_height"
    resolution = 10

    script_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(script_dir, "handovers-fig4a+10b.csv"))
    durations = pd.read_csv(os.path.join(script_dir, "handover-durations-fig4a+10b.csv"))
    durations = durations.set_index("Altitude")
    data = data.fillna(0)
    data[corr_param] = data[corr_param].apply(lambda x: custom_round(x, base=resolution))
    ho_all = pd.DataFrame()
    for i in range(1,7):
        ho = data[data["pi"] == i].groupby([corr_param, "flight_ts"])["ho"].count()
        ho = ho.sort_index()
        for j in range(len(ho.keys())):
            ho_keys = ho.keys()[j]
            for durations_keys in durations.keys():
                if durations_keys in ho_keys[1]:
                    ho.loc[ho_keys] = ho.loc[ho_keys] / durations[durations_keys][ho_keys[0]]
        ho_all = pd.concat([ho_all, ho])
    ho_all.columns = ["values"]
    ho_all = ho_all.reset_index()
    addr = ['Altitude', 'Flight']
    ho_all = ho_all[['values']].join(ho_all["index"].apply(lambda loc: pd.Series(loc, index=addr)))
    
    newcols = {"GroundOrAir": [], "location": [], "day": [], "operator": []}
    for index, row in ho_all.iterrows():
        if row["Altitude"] < 10:
            newcols["GroundOrAir"].append("Ground")
        else:
            newcols["GroundOrAir"].append("Air")
        location, day, operator = row["Flight"].split("-")
        newcols["location"].append(location)
        newcols["day"].append(day)
        newcols["operator"].append(operator)
    
    ho_all["GroundOrAir"] = newcols["GroundOrAir"]
    ho_all["location"] = newcols["location"]
    ho_all["day"] = newcols["day"]
    ho_all["operator"] = newcols["operator"]

        
    rural = [ho_all["values"][(ho_all["operator"] == "P1") & (ho_all["GroundOrAir"] == "Air")],
            ho_all["values"][(ho_all["operator"] == "P1") & (ho_all["GroundOrAir"] == "Ground")]]
    urban = [ho_all["values"][(ho_all["operator"] == "P2") & (ho_all["GroundOrAir"] == "Air")],
            ho_all["values"][(ho_all["operator"] == "P2") & (ho_all["GroundOrAir"] == "Ground")]]

    util.use_style()
    figsize = (util.columnwidth_acmart, 1.2)
    ticks = ['Air', 'Grd']
    fig1, ax1 = plt.subplots(figsize=figsize)

    rural_plot = ax1.boxplot(rural,positions=np.array(np.arange(len(rural)))*.03+0.0065, 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.009,
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))

    urban_plot = ax1.boxplot(urban, positions=np.array(np.arange(len(urban)))*.03-0.0065, 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.009, 
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))
    
    set_boxplot_color(urban_plot, util.mpl_colors()[1], 'P2')
    set_boxplot_color(rural_plot, util.mpl_colors()[0], 'P1')
    text(0.67, 0.04,'\\textbf{(b)}')

    # set the x label values
    plt.grid(axis='y')
    plt.xlim([-0.005, 0.72])
    plt.yticks([0, 0.03], ticks, rotation=90)
    plt.ylim([-0.0175, 0.0495])
    plt.ylabel("")
    plt.xscale("linear")
    plt.xticks(np.arange(0, 0.72, 0.1))
    plt.xlabel("Frequency (Handover/s)")
    util.set_actual_figsize(fig1, figsize)
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fig1.savefig(args.save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

if __name__ == '__main__':
    main()
