#!/usr/bin/env python3

import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sqlite3
import pandas as pd
import numpy as np
import os
import roq_individual as roq
import util

from matplotlib.legend_handler import HandlerLine2D
# https://stackoverflow.com/a/48308814
def update_legend_line(handle, orig):
    handle.update_from(orig)
    handle.set_linewidth(2)

figsize = (util.columnwidth_acmart, util.columnwidth_acmart/1.6)
linestyles = ['solid', 'dotted', 'dashed']

def plot_cdf_groupby(ax, labels, dfs, groupby, accessor_fn):
    parts = []
    for i, (label, df) in enumerate(zip(labels, dfs)):
        if groupby is None:
            data = accessor_fn(df)
            parts.append(util.plot_cdf(ax, data, label=label)[0])
        else:
            for j, (groupby_label, groupby_idxs) in enumerate(groupby):
                df_lens = df[df['index_id'].isin(groupby_idxs)]
                data = accessor_fn(df_lens)
                line_label = f"{label} - {groupby_label}"
                parts.append(util.plot_cdf(ax, data, label=line_label,
                                           color=util.mpl_colors()[i],
                                           linestyle=linestyles[j])[0])
    return parts


def plot_boxplot_groupby_vert(ax, labels, dfs, groupby, accessor_fn):
    '''
    groupby=list of tuples of label and list of index_ids
    '''
    # labels = list(reversed(labels))
    # dfs = list(reversed(dfs))
    # groupby = list(reversed(groupby))

    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', visible=True)
    bps = []
    if groupby is None or len(groupby) == 1:
        data = [accessor_fn(df) for df in dfs]
        bps.append(ax.boxplot(data, labels=labels, showfliers=False, vert=True,
                              boxprops=dict(facecolor=util.mpl_colors()[0])))
    else:
        if len(groupby) > 2:
            raise Exception("Grouped boxplot not implemented for more than 2 boxplots per tick")
        width_bp = 0.6
        margin_bp = 0.1
        group_sep = 1.2
        for i, groupby_idxs in enumerate([v[1] for v in groupby]):
            data = []
            for df in dfs:
                df_lens = df[df['index_id'].isin(groupby_idxs)]
                data.append(accessor_fn(df_lens))
            offset = -0.4 + 0.8 * i
            # offset = -(width_bp + margin_bp)*len(groupby) + (width_bp + margin_bp)*i
            # offset = (width_bp + margin_bp)/2 + (width_bp + margin_bp) * idx_loc
            ypos = np.arange(len(labels))*2 + offset
            for label, d in zip(labels, data):
                print(f"Goodput {groupby[i][0]}-{label}:\tmean={d.mean():.2f}, median={d.median():.2f} Mbps")
            bps.append(ax.boxplot(
                data, showfliers=False, vert=False, positions=ypos, widths=width_bp,
                boxprops=dict(facecolor=util.mpl_colors()[i]), patch_artist=True))
        ax.set_yticks(np.arange(0, len(labels)*2, 2), labels,
                      size=plt.rcParams['axes.labelsize'], rotation=90,
                      alpha=0)

        ax.text(-0.04, 0.03, labels[0], transform=ax.transAxes, size=10, rotation=90)
        ax.text(-0.04, 0.33, labels[1], transform=ax.transAxes, size=10, rotation=90)
        ax.text(-0.04, 0.78, labels[2], transform=ax.transAxes, size=10, rotation=90)

        ax.set_ylim(-1, len(labels)*2-1)

    if groupby is not None and len(bps) > 0:
        ax.legend(reversed([bp['boxes'][0] for bp in bps]), reversed([v[0] for v in groupby]), loc='lower right')

def plot_playback_latency_cdf(labels, df_timings, groupby=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    parts = plot_cdf_groupby(ax, labels, df_timings, groupby,
                             accessor_fn=roq.get_playback_latency)
    parts.append(ax.axvline(300, color=util.threshold_color, alpha=1, zorder=0,
                            label='Threshold', linestyle='dashed'))
    ax.set_ylabel("CDF")
    ax.set_xlabel("Playback Latency (ms)")
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=1000)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    # ax.legend(loc='lower right')
    # for part in parts:
    #     if part.get_label() == 'Static - Urban':
    #         part.set_label("Static - Urban (25 Mbps)")
    #     elif part.get_label() == 'Static - Rural':
    #         part.set_label("Static - Rural (8 Mbps)")
    legend_figsize = ax.get_figure().get_size_inches()
    legend_figsize = (legend_figsize[0], 1)
    fig_legend1 = util.plot_external_legend(parts, legend_figsize,
        ncol=4, borderpad=.4, mode='expand',
        handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})
    fig_legend2 = util.plot_external_legend(parts, legend_figsize,
        ncol=4, borderpad=.4, mode='expand',
        columnspacing=0.8, handlelength=0.8, handletextpad=0.2,
        handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})
    # fig_legend2 = util.plot_external_legend(parts, legend_figsize,
    #     ncol=len(parts), columnspacing=0.8, handlelength=1.0, handletextpad=0.2,
    #     frameon=False,
    #     handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})
    return fig, fig_legend1, fig_legend2

def plot_fps_cdf(labels, df_timings, log_y=True, groupby=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    ax.set_ylabel("CDF")
    ax.set_xlabel("Frames per second")
    dfs = [roq.get_timing_based_fps(df, reset_index=False).reset_index()
                for df in df_timings]
    # for label, fps in zip(labels, fps_data):
    #     util.plot_cdf(ax, fps, label=label)
    parts = plot_cdf_groupby(ax, labels, dfs, groupby,
                             accessor_fn=lambda df: df['fps'])
    if log_y:
        ax.set_yscale('log')
        # ax.set_ylim(bottom=None, top=1.5)
        ax.set_ylim(bottom=0.001, top=1)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    else:
        ax.set_ylim(bottom=0, top=1.01)
    # ax.set_xlim(left=-3, right=None)
    ax.set_xlim(left=0, right=40)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    yminticks = list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1, 0.1))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(yminticks))
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.0001, 0.001, 0.01, 0.1, 1]))

    # ax.legend(loc='lower right')
    fig_legend = util.plot_external_legend(
        parts, (util.columnwidth_acmart*2, 1), ncol=len(parts), # figsize, ncol=4,
        columnspacing=0.8, handlelength=1.0, handletextpad=0.2, frameon=False,
        handler_map={plt.Line2D: HandlerLine2D(update_func=update_legend_line)})
    return fig, fig_legend

def plot_goodput_boxplot(labels, df_ps, groupby=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    plot_boxplot_groupby_vert(ax, labels, df_ps, groupby, roq.get_throughput)
    ax.set_xlabel("Goodput (Mbps)")
    ax.set_xlim(left=0)
    return fig

def plot_ssim_cdf(labels, df_ssims, groupby=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    ax.set_ylabel("CDF")
    ax.set_xlabel("SSIM")
    parts = plot_cdf_groupby(ax, labels, df_ssims, groupby,
                             accessor_fn=lambda df: df['ssim'])
    for i, (label, df) in enumerate(zip(labels, df_ssims)):
        for j, (groupby_label, groupby_idxs) in enumerate(groupby):
            line_label = f"{label} - {groupby_label}"
            df_lens = df[df['index_id'].isin(groupby_idxs)]
            data = df_lens['ssim']
            perc = sum(data < 0.5) / len(data) if len(data) > 0 else 0
            print(f"{line_label}: {perc}")

    parts.append(ax.axvline(0.5, color=util.threshold_color, zorder=1,
                            label='threshold', linestyle='dashed'))
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=1)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.9g}'))
    # fig.tight_layout()
    if len(parts) <= 4:
        ax.legend(loc='upper left')
        return [fig]
    else:
        fig_legend = util.plot_external_legend(parts, figsize,
                                               # ncol=4,
                                               ncol=8,
                                               columnspacing=.8,
                                               handlelength=1.0,
                                               handletextpad=0.2)
        fig_legend.tight_layout()
        return fig, fig_legend

def get_df_timing_data(conn, ccs, base_selector):
    df_timings = []
    base_tss = dict() # starts with first played frame which is later than the "real" start but good enough for ssim timestamps
    for cc in ccs:
        idxs = base_selector & util.ids_with(conn, 'type', cc)
        if cc == 'static':
            # idxs &= util.ids_with(conn, 'notes', static_bw)
            idxs &= util.ids_with(conn, 'notes', '25 Mbps') | util.ids_with(conn, 'notes', '8 Mbps')
        df_timing = pd.read_sql(f"SELECT * FROM timing WHERE index_id IN ({util.in_clause_values(idxs)})",
                                conn, parse_dates=['ts'], index_col='ts')
        air_filter = util.get_air_filter(conn, idxs, df_timing.index)
        df_timing_air = df_timing[air_filter]
        if not df_timing_air.empty:
            df_timings.append(df_timing_air)
            base_tss[cc] = df_timing_air.index[0]
        else:
            print(f"No timing data found for {cc=}")
    return df_timings, base_tss

def get_df_ps(conn, ccs, base_selector):
    df_ps = []
    for cc in ccs:
        selector = base_selector & util.ids_with(conn, 'type', cc)
        if cc == 'static':
            selector &= util.ids_with(conn, 'notes', '25 Mbps') | util.ids_with(conn, 'notes', '8 Mbps')
        idxs_p = selector & util.ids_with(conn, 'recorded_on', 'server')
        print(f"Query RTP: {cc}")
        df_p = pd.read_sql(f"SELECT ts, index_id, size, seqnum_unwr FROM rtp WHERE index_id IN ({util.in_clause_values(idxs_p)})",
                        conn, parse_dates=['ts'], index_col=['ts'])
        df_ps.append(df_p)
    return df_ps

def get_df_ssims(conn, ccs, base_selector, base_tss):
    df_ssims = []
    for cc in ccs:
        idxs = base_selector & util.ids_with(conn, 'type', cc) & util.ids_with(conn, 'recorded_on', 'server')
        if cc == 'static':
            # idxs &= util.ids_with(conn, 'notes', static_bw)
            idxs &= util.ids_with(conn, 'notes', '25 Mbps') | util.ids_with(conn, 'notes', '8 Mbps')
        print(f"Query SSIM: {cc}")
        df_ssim = pd.read_sql(f"SELECT index_id, frame_streamer, ssim FROM ssim WHERE index_id IN ({util.in_clause_values(idxs)}) AND frame_streamer>30",
                            conn) # skip initial synchronization phase
        df_ssim['ts'] = base_tss[cc] + pd.to_timedelta(df_ssim['frame_streamer'] / 30, unit='s')
        air_filter = util.get_air_filter(conn, idxs, df_ssim['ts'])
        df_ssims.append(df_ssim)
    return df_ssims

def main_figure_6_7(args):
    util.use_style()
    conn = sqlite3.connect(args.database)

    days = ['220313U', '220323R']
    ccs = ['static', 'scream', 'gcc']
    providers = ['P1']
    groupby = [('Urban', util.ids_with(conn, 'day', days[0])),
               ('Rural', util.ids_with(conn, 'day', days[1]))]

    figsize=(util.textwidth_acmart, 2)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize) # CDFs
    figsize_gput=(util.columnwidth_acmart, 1.6)
    fig_gput, ax_gput = plt.subplots(figsize=figsize) # goodput boxplot

    figs_legend = []

    base_selector = util.ids_with(conn, 'day', days) & util.ids_with(conn, 'provider', providers)
    df_timings, base_tss = get_df_timing_data(conn, ccs, base_selector)
    df_ps = get_df_ps(conn, ccs, base_selector)
    df_ssims = get_df_ssims(conn, ccs, base_selector, base_tss)

    cc_labels = [util.capitalize_cc_name(cc) for cc in ccs]

    _, fig_legend0 = plot_fps_cdf(cc_labels, df_timings, log_y=True, groupby=groupby, ax=axes.flat[0])
    plot_ssim_cdf(cc_labels, df_ssims, groupby=groupby, ax=axes.flat[1])
    _, fig_legend1, fig_legend2 = plot_playback_latency_cdf(cc_labels, df_timings, groupby=groupby, ax=axes.flat[2])
    figs_legend = [fig_legend1, fig_legend2]#, fig_legend1]

    plot_goodput_boxplot(cc_labels, df_ps, groupby=groupby, ax=ax_gput)

    labels = [r'\textbf{' + f"({v})" + r'}' for v in ['a', 'b', 'c', 'd']]
    for i, ax in enumerate(axes.flat):
        ax.text(0.025, 0.90,# 0.025, 0.85,
                labels[i], # (x, y)
                transform=ax.transAxes, size=10, weight='bold')

    axes.flat[1].set_ylabel("")
    axes.flat[2].set_ylabel("")

    fig.subplots_adjust(wspace=0.2) # reduce horizontal space

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with PdfPages(args.save) as pdf:
            for fig, figsize in [(fig, figsize), (fig_gput, figsize_gput)]:
                success = util.set_actual_figsize(fig, figsize)
                if success is not True:
                    print(f"Forcing exact figsize: {success}")
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            for fig in figs_legend:
                pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.show()

def main_figure_12(args):
    util.use_style()
    conn = sqlite3.connect(args.database)

    days = ['220323R']
    ccs = ['static', 'scream', 'gcc']
    cc_labels = [util.capitalize_cc_name(cc) for cc in ccs]
    providers = ['P1', 'P2']
    groupby = [(util.provider_pseudonym(provider), util.ids_with(conn, 'provider', provider))
                for provider in providers]

    figsize=(util.textwidth_acmart, 3.2)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize) # goodput boxplots and FPS CDFs

    figs_legend = []

    day = days[0]
    print(f"Query {day}")
    base_selector = util.ids_with(conn, 'day', day) & util.ids_with(conn, 'provider', providers)
    df_timings, base_tss = get_df_timing_data(conn, ccs, base_selector)
    df_ps = get_df_ps(conn, ccs, base_selector)
    df_ssims = get_df_ssims(conn, ccs, base_selector, base_tss)

    plot_goodput_boxplot(cc_labels, df_ps, groupby=groupby, ax=axes.flat[0])
    _, fig_legend0 = plot_fps_cdf(cc_labels, df_timings, log_y=True, groupby=groupby, ax=axes.flat[1])

    _, fig_legend1, fig_legend2 = plot_playback_latency_cdf(cc_labels, df_timings, groupby=groupby, ax=axes.flat[2])
    _, fig_legend3 = plot_ssim_cdf(cc_labels, df_ssims, groupby=groupby, ax=axes.flat[3])

    figs_legend = [fig_legend1]

    labels = [r'\textbf{' + f"({v})" + r'}' for v in ['a', 'b', 'c', 'd']]
    for i, ax in enumerate(axes.flat):
        ax.text(0.025, 0.87, labels[i], # (x, y)
                transform=ax.transAxes, size=10, weight='bold')

    fig.subplots_adjust(hspace=0.3) # increase vertical space between plots
    fig.subplots_adjust(wspace=0.2) # reduce horizontal space between the figs

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with PdfPages(args.save) as pdf:
            success = util.set_actual_figsize(fig, figsize)
            if success is not True:
                print(f"Forcing exact figsize: {success}")
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            for fig in figs_legend:
                pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot CDF/boxplot video metric figures")
    parser.add_argument("database", help="The sqlite database file")
    parser.add_argument("figure", help="Choose 'urban' to plot figures 6&7 or 'rural' to plot figure 12", choices=["urban", "rural"])
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()
    if args.figure == "urban":
        main_figure_6_7(args)
    if args.figure == "rural":
        main_figure_12(args)

if __name__ == "__main__":
    main()
