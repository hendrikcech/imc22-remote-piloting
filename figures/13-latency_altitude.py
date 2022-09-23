#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import sqlite3
import pandas as pd
import util

def get_roq_latency(conn, idxs):
    sql = '''
SELECT t2.index_id, t2.ts_sent, t2.latency_ms, t1.relative_height
FROM latency AS t2
JOIN flight AS t1
ON t1.rowid = (SELECT rowid FROM flight AS t3 WHERE t3.ts <= t2.ts_sent ORDER BY ts DESC, offset_time DESC LIMIT 1)
WHERE t1.relative_height IS NOT NULL
    '''
    sql += f" AND index_id IN ({util.in_clause_values(idxs)})"
    df = pd.read_sql(sql, conn, parse_dates=['ts_sent'])
    return df

def get_ping_latency(conn, idxs):
    sql = '''
SELECT t2.index_id, t2.ts, t2.rtt AS latency_ms, t1.relative_height
FROM ping AS t2
JOIN flight AS t1
ON t1.rowid = (SELECT rowid FROM flight AS t3 WHERE t3.ts <= t2.ts ORDER BY ts DESC, offset_time DESC LIMIT 1)
WHERE t1.relative_height IS NOT NULL
    '''
    sql += f" AND index_id IN ({util.in_clause_values(idxs)})"
    df = pd.read_sql(sql, conn, parse_dates=['ts_sent'])
    return df

# Plot contains 4 lines: ground/air rural/urban
def plot_altitude_05_cdf(df, data_source, plot_label, xscale=None):
    buckets_altitude = [(0, 20), (21, 60), (61, 100), (101, 140)]
    data_altitude = [df[(df['relative_height'] >= l) & (df['relative_height'] <= u)]['latency_ms']
                        for l, u in buckets_altitude]
    labels_altitude = [f"{l}–{u} m" for l, u in buckets_altitude]
    parts = []
    # print(f"Min latencies:{[d['latency_ms'].min() for (d, l) in data]}")
    def plot_cdf(ax):
        for i_bucket, (data, label) in enumerate(zip(data_altitude, labels_altitude)):
            color = util.mpl_colors()[i_bucket]
            parts.extend(util.plot_cdf(ax, data, label=label, color=color))
    def plot_avg(ax):
        for i_bucket, (data, label) in enumerate(zip(data_altitude, labels_altitude)):
            color = util.mpl_colors()[i_bucket]
            parts.append(ax.axvline(data.mean(),
                                    # label=f"{label} Average",
                                    color=color, linestyle='dotted', alpha=0.8, zorder=1))
    def style(ax, left_xlim=10, right_xlim=None):
        ax.grid(axis='x')
        # ax.set_ylim(bottom=0, top=1.008)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0, right=None)
        if xscale == 'symlog': # log scale
            linthresh = 500
            ax.set_xscale('symlog', linthresh=linthresh)
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
            major_locators = list(range(0, linthresh, 100)) + [500, 1000, 2000, 4000]
            ax.xaxis.set_major_locator(mticker.FixedLocator(major_locators))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        elif xscale == 'log':
            ax.set_xlim(left=left_xlim, right=right_xlim)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
            major_locators = [left_xlim, 100, 1000, 10000]
            if right_xlim is not None:
                major_locators = sorted(major_locators + [right_xlim])
            minor_locators = list(range(0, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
            ax.xaxis.set_major_locator(mticker.FixedLocator(major_locators))
            ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_locators))
    main_fig, main_ax = plt.subplots(figsize=figsize)
    plot_cdf(main_ax)
    plot_avg(main_ax)
    style(main_ax, left_xlim=20)

    # https://stackoverflow.com/a/48308814
    def update_legend_line(handle, orig):
        handle.update_from(orig)
        handle.set_linewidth(6)

    fig_legend = util.plot_external_legend(parts, figsize, ncol=4,
                                           # columnspacing=1,
                                           # handlelength=0.2,
                                           # handletextpad=0.6,
                                           borderpad=0.6,
                                           frameon=True,
                                           handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})
    in_ax = inset_axes(main_ax, width="100%", height="100%",
                       bbox_transform=main_ax.transAxes,
                        # (0,0) is bottom left (x,y). last two values are size.
                       bbox_to_anchor=(.48, .26, .5, .6))
    # plt.setp(in_ax.spines.values(), linewidth=0.6)
    plot_cdf(in_ax)
    plot_avg(in_ax)
    mark_inset(main_ax, in_ax, loc1=1, loc2=3, ec="0.5", fc="none", linewidth=0.6)
    style(in_ax, left_xlim=40, right_xlim=500)

    in_ax.set_ylim(bottom=0.95, top=1)
    in_ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    in_ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    # style(main_ax)

    # main_ax.set_ylabel("p [one-way delay ≤ t]")
    # main_ax.set_xlabel("ms")
    # main_ax.set_ylabel("Cumulative Distribution")
    main_ax.set_ylabel("CDF")
    if data_source == 'rtp' or data_source == 'ping_owd':
        main_ax.set_xlabel("One-Way Latency (ms)")
    elif data_source == 'ping_rtt':
        main_ax.set_xlabel("RTT (ms)")
    else:
        raise Exception("Invalid source")

    util.set_actual_figsize(main_fig, figsize)
    main_ax.text(0.03, 0.87, r'\textbf{' + plot_label + '}', transform=main_ax.transAxes, size=10)
    return main_fig, fig_legend

def desc(series):
    return series.describe().apply(lambda x: format(x, 'f'))

# def print_latency_summary(lat_grd, lat_air):
#     desc_grd = desc(lat_grd['latency_ms'])
#     desc_air = desc(lat_air['latency_ms'])
#     print(f"Ground:\n{desc_grd}Air:\n{desc_air}")

figsize = (util.columnwidth_acmart, 2)

def main():
    global base_ts, end_ts
    parser = argparse.ArgumentParser(description="Compare metrics in ground and air tests")
    parser.add_argument("database", help="Database")
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()

    util.use_style()

    conn = sqlite3.connect(args.database)

    # idxs_grd = util.ids_with(conn, 'day', '220205U')
    idxs_urban = util.ids_with(conn, 'day', '220205U')
    idxs_rural = util.ids_with(conn, 'day', '220203R')
    # idxs_grd  = util.ids_with(conn, 'day', '220205U') & util.ids_with(conn, 'flight', [16, 17, 18, 19])
    # idxs_grd |= util.ids_with(conn, 'day', '220203R') & util.ids_with(conn, 'flight', [8, 14, 15])
    idxs_air  = util.ids_with(conn, 'day', '220205U') & util.ids_with(conn, 'flight', [3,4,5,6,7,8, 10,11,12,13,14,15])
    idxs_air |= util.ids_with(conn, 'day', '220203R') & util.ids_with(conn, 'flight', [1,2, 4,5,6,7, 9,10,11,12,13])
    # idxs_air  = util.ids_with(conn, 'day', '220205U')
    # idxs_air |= util.ids_with(conn, 'day', '220203R')
    idxs_rtp = util.ids_with(conn, 'type', ['gcc', 'scream', 'static'])

    figs = []

    print("Retrieving latency df")
    # lat = get_roq_latency(conn, idxs_air & idxs_rtp)
    lat = get_ping_latency(conn, idxs_air)
    # print_latency_summary(lat_grd, lat_air)
    lat_urban = lat[lat['index_id'].isin(idxs_urban)]
    lat_rural = lat[lat['index_id'].isin(idxs_rural)]
    # print(lat_urban)
    # print(lat_urban[lat_urban['relative_height'] > 0])
    # print(lat_rural)
    # print(lat_rural[lat_rural['relative_height'] > 0])

    print("Retrieving latency df")
    # figs.extend(plot_altitude_05_cdf(lat_urban, 'rtp', xscale='log'))
    # figs.extend(plot_altitude_05_cdf(lat_rural, 'rtp', xscale='log'))
    figs.extend(plot_altitude_05_cdf(lat_urban, 'ping_rtt', '(a)', xscale='log'))
    figs.extend(plot_altitude_05_cdf(lat_rural, 'ping_rtt', '(b)', xscale='log'))
    # figs.append(plot_altitude_cdf(lat_rural))

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with PdfPages(args.save) as pdf:
            pdf.savefig(figs[0], bbox_inches='tight', pad_inches=0)
            pdf.savefig(figs[1], bbox_inches='tight')
            pdf.savefig(figs[2], bbox_inches='tight', pad_inches=0)
            pdf.savefig(figs[3], bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    main()
