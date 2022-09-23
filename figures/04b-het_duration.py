#!/usr/bin/env python3

import util
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.legend_handler import HandlerLine2D
import argparse
import os
import util


# Taken from https://stackoverflow.com/a/20132614
def set_boxplot_color(bp, color):
    for attr in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(bp[attr], color=color)

def mark_hos_inair(conn, df_ho, air_thresh=1):
    df_flight = pd.read_sql("SELECT ts, relative_height FROM flight ORDER BY ts", conn, parse_dates=['ts'])
    df_inair = df_flight[df_flight['relative_height'] > air_thresh].copy()
    df_inair['block'] = df_inair['ts'].diff().dt.seconds.gt(10).cumsum()
    inair_blocks = df_inair.groupby('block').agg({'ts': ['first', 'last']})
    inair_blocks.columns = ['start', 'end']
    df_ho_inair_column = [
        ((inair_blocks['start'] <= ts) & (inair_blocks['end'] >= ts)).any()
        for ts in df_ho['start_ts'].array]
    df_ho['inair'] = df_ho_inair_column
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax.plot(df_flight['ts'], df_flight['relative_height'], label='height')
    # ax2.plot(df_inair['ts'], df_inair['block'], label='block', color='orange',
    #          marker='x')
    # plt.legend()
    # plt.show()
    return df_ho


def main():
    parser = argparse.ArgumentParser(description="Plot the HET in a boxplot")
    parser.add_argument('database', help="The sqlite database file")
    parser.add_argument('--save', help="Save as PDF")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    df_ho = pd.read_sql("SELECT start_ts, end_ts, duration FROM handovers ORDER BY start_ts", conn, parse_dates=['start_ts', 'end_ts'])
    df_ho['start_ts'] = df_ho['start_ts'].dt.tz_localize('UTC')
    df_ho['end_ts'] = df_ho['end_ts'].dt.tz_localize('UTC')
    days = {
        # Don't include '2022-02-12': flight logs are missing
        'rural': ['2021-05-26', '2021-10-26', '2022-02-03', '2022-03-23'],
        'urban': ['2021-05-24', '2021-10-10', '2022-02-05', '2022-03-13']
        # 'urban': ['2021-05-24'],
        # 'rural': ['2021-05-26']
    }
    for key in days.keys():
        days[key] = [pd.to_datetime(v, utc=True) for v in days[key]]

    # Add inair boolean column to df_ho
    mark_hos_inair(conn, df_ho)

    # Split df_ho into urban and rural
    df_locs = { location: df_ho[df_ho['start_ts'].dt.floor('D').isin(days[location])]
                for location in days.keys() }

    figsize=(util.columnwidth_acmart, 1.2)
    util.use_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis='y', visible=False)

    positions = ['Air', 'Grd']
    width_bp = 0.4
    margin_bp = 0.1
    group_sep = 1.2
    for idx_pos, label_pos in enumerate(positions):
        for idx_loc, (label_loc, df) in enumerate(df_locs.items()):
            offset = (width_bp + margin_bp)/2 + (width_bp + margin_bp) * idx_loc
            pos = idx_pos * group_sep + offset
            inair_value = True if label_pos == 'Air' else False
            het_ms = df.loc[df['inair'] == inair_value, 'duration'] * 1000
            print(f"{label_pos},{label_loc},{label_pos} ({pos=}): mean={het_ms.mean():.2f}, median={het_ms.median():.2f}, std={het_ms.std():.2f}, len={het_ms.size}")
            bp = ax.boxplot(het_ms.array, vert=False,
                            boxprops=dict(facecolor=util.mpl_colors()[idx_loc]),
                            positions=[pos], widths=width_bp)
    ticks_one = (width_bp+margin_bp)*2 / 2
    ticks_two = ticks_one + group_sep
    ax.set_yticks([ticks_one, ticks_two], positions, rotation=90,
                  size=plt.rcParams['axes.labelsize'])
        # ax.set_xlim(-1, len(labels)*2-1)

    parts = [ax.plot([], c=util.mpl_colors()[i], label=label.capitalize())[0]
                for i, label in enumerate(df_locs.keys())]
    # ax.legend(ncol=len(df_locs), loc='center right')
    # ax.legend(ncol=2, bbox_to_anchor=(0,0))
    # https://stackoverflow.com/a/48308814
    def update_legend_line(handle, orig):
        handle.update_from(orig)
        handle.set_linewidth(8)
    fig_legend = util.plot_external_legend(parts, figsize, ncol=2,
                                           columnspacing=1,
                                           handlelength=0.2,
                                           handletextpad=0.6,
                                           frameon=False,
                                           handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})

    ax.set_xlim(left=0.1, right=None)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.set_xlabel('HET duration (ms)')

    ax.axvline(50, color=util.threshold_color, linestyle='dashed', zorder=1)


    ax.text(0.92, 0.83, r'\textbf{(b)}', transform=ax.transAxes, size=10)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        print(util.set_actual_figsize(fig, figsize))
        with PdfPages(args.save) as pdf:
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            bbox = fig_legend.get_tightbbox(fig_legend.canvas.get_renderer())
            print(bbox)
            bbox_mut = bbox.get_points()
            bbox_mut[0][0] -= 0.2
            bbox_mut[1][1] += 0.2
            pdf.savefig(fig_legend, bbox_inches='tight')#, pad_inches=0, bbox_inches=Bbox(bbox_mut))
                        # bbox_inches=Bbox([[0, 0], [util.columnwidth_acmart, 0.40]]))
    else:
        plt.show()



if __name__ == '__main__':
    main()



