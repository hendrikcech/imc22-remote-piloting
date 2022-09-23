import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import text

import roq_individual as roq
import util
pd.options.mode.chained_assignment = None

def convert_time_to_seconds(data):
    data['ts_secs'] = (data['ts'] - data['ts'][0]).dt.total_seconds()
    data['ts_secs'].fillna(0, inplace=True)
    return data

def fetch_data_for_timeplots(conn):
    pi=5
    c = conn.cursor()
    c.execute("SELECT id FROM `index` WHERE day='220323R' AND flight=4 AND pi=5 AND recorded_on='pi'")
    index_s = c.fetchone()[0]
    c.execute("SELECT id FROM `index` WHERE day='220323R' AND flight=4 AND pi=5 AND recorded_on='server'")
    index_p = c.fetchone()[0]

    df_all = {}
    
    df_all["cc"] = pd.read_sql(f"SELECT * FROM gcc WHERE index_id='{index_s}' ORDER BY ts",
                        conn, parse_dates=['ts'], index_col='ts')
    df_all["timing"] = pd.read_sql(f"SELECT * FROM timing WHERE index_id='{index_p}' ORDER BY ts",
                                conn, parse_dates=['ts'], index_col='ts')
    df_all["rtp-server"] = pd.read_sql(f"SELECT * FROM rtp WHERE index_id='{index_s}' ORDER BY ts", 
                    conn, parse_dates=['ts'])
    df_all["rtp-player"] = pd.read_sql(f"SELECT * FROM rtp WHERE index_id='{index_p}' ORDER BY ts", 
                    conn, parse_dates=['ts'])
    df_all["tssim"] = pd.read_sql(f"SELECT id, frame_streamer, frame_player, ssim FROM ssim WHERE index_id='{index_p}' ORDER BY id", conn)
    df_all["flights"] = pd.read_sql(f"SELECT * FROM flight ORDER BY ts", 
                          conn, parse_dates=['ts'])
    df_all["ho"] = pd.read_sql(f"SELECT * FROM handovers WHERE pi='{pi}' ORDER BY start_ts",
                        conn, parse_dates=['start_ts', 'end_ts'])

    base_ts = df_all["rtp-server"].loc[0, 'ts']
    if len(df_all["tssim"]) > 0:
        key = 'frame_streamer'
        df_ssim = df_all["tssim"]
        df_ssim['ts'] = base_ts + pd.to_timedelta(df_ssim[key] / 30, unit='s')

    df_all["ho"]['start_ts'] = df_all["ho"]['start_ts'].dt.tz_localize('UTC')
    df_all["ho"]['end_ts'] = df_all["ho"]['end_ts'].dt.tz_localize('UTC')

    info = roq.get_info(conn, index_s)

    if info is None:
        breakpoint()
    air_time_df_cc = (df_all["cc"].index <= info['air_ts'][1])
    air_time_df_timing = (df_all["timing"].index <= info['air_ts'][1])
    air_time_df_rtp_s = (df_all["rtp-server"]["ts"] <= info['air_ts'][1])
    air_time_df_rtp_p = (df_all["rtp-player"]["ts"] <= info['air_ts'][1])
    air_time_df_ssim = (df_ssim["ts"] <= info['air_ts'][1])
    air_time_df_ho = (df_all["ho"]["start_ts"] >= info['air_ts'][0]) & (df_all["ho"]["end_ts"] <= info['air_ts'][1])
    air_time_df_flights = (df_all["flights"]["ts"] >= df_all["rtp-server"]["ts"][0]) & (df_all["flights"]["ts"] <= info['air_ts'][1])
    
    df_all["rtp-server"] = df_all["rtp-server"][air_time_df_rtp_s]
    df_all["rtp-player"] = df_all["rtp-player"][air_time_df_rtp_p]
    df_all["rtp"] = pd.merge(df_all["rtp-server"], df_all["rtp-player"], how='left', on='seqnum_unwr', suffixes=('_s', '_p'))
    df_all["cc"] = df_all["cc"][air_time_df_cc]
    df_all["timing"] = df_all["timing"][air_time_df_timing]
    df_all["ssim"] = df_ssim[air_time_df_ssim]
    df_all["flights"] = df_all["flights"][air_time_df_flights].reset_index()
    df_all["ho"] = df_all["ho"][air_time_df_ho]
    return df_all

def get_fps_pblatency(data):
    data_fps = roq.get_timing_based_fps(data)
    data_pblatency = roq.get_playback_latency(data)
    return data_fps, data_pblatency

def plot_latency(df_all):
    figsize = (util.columnwidth_acmart, 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    p1 = sns.lineplot(x=roq.rel_index(df_all["network_latency"].index), y=df_all["network_latency"],
                    data=df_all["network_latency"], linewidth=1, alpha=1, color="steelblue",
                    zorder=5)


    p2 = sns.lineplot(x=roq.rel_index(df_all["pblatency"].index), y=df_all["pblatency"],
                    data=df_all["pblatency"], linewidth=2, alpha=1, color="sandybrown",
                    zorder=3)
    
    p3 = roq.handovers(ax, df_all["ho"], df_all["network_latency"].index[0])
    ax2 = p1.axes.twinx()
    p3 = ax2.scatter(roq.rel_index(df_all["network_loss"].index), df_all["network_loss"] * 100, 
               c='r', marker='D', s=15)
    p1.set(xlabel='Flight Time (s)', ylabel='')
    ax2.set_ylabel("Packet Loss (\%)")   
    plt.xticks(np.arange(0, 200, 25))
    ax.set_xlim([-4, 195])
    ax.set_yticks(np.arange(0, 801, 200))

    p4 = ax.axvspan(xmin=83.6, xmax=83.7, color='dimgrey', alpha=1, zorder=0)
    p4 = ax.axvspan(xmin=82.6, xmax=83.6, color='yellow', alpha=0.4, zorder=0)
    p4 = ax.axvspan(xmin=83.7, xmax=84.7, color='green', alpha=0.4, zorder=0)
    p1.set(xlabel='Flight Time (s)', ylabel='Latency (ms)')
    text(81.25, 800,'\\textbf{(a)}')
    plt.axis([81, 86, 0, 950])

    ax.set_ylim([0, 900])
    plt.tight_layout()
    util.set_actual_figsize(fig, figsize)
    return fig

def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def losses(df):
    cutoff = (df['ts_s'].iloc[-1]) - pd.Timedelta('10s')
    df_lens = df[df['ts_s'] < cutoff]
    df_net_loss = roq.get_network_losses(df_lens)
    df_snd_loss = roq.get_sender_losses(df_lens)
    all_sec = df_lens.resample('1s', on='ts_s').count()['index_id_s']
    lost_perc, lost_sec = None, None
    if not df_net_loss.empty:
        lost_sec = df_net_loss.resample('1s', on='ts_s').count()['ts_s']
        lost_perc = (lost_sec / all_sec).fillna(0)
        lost_perc = lost_perc[lost_perc > 0]
    if not df_snd_loss.empty:
        lost_sec = df_snd_loss.resample('1s', on='ts_s').count()['ts_s']
        lost_sec = lost_sec[lost_sec > 0]

    return lost_perc, lost_sec

def main():
    parser = argparse.ArgumentParser(description="Plot latencies and handover occurence")
    parser.add_argument("database", help="Database")
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    util.use_style()

    df_all = fetch_data_for_timeplots(conn)
    tput_s = roq.get_throughput(df_all["rtp-server"].set_index('ts'))
    tput_p = roq.get_throughput(df_all["rtp-player"].set_index('ts'))
    df_all["cc"]["target"] = df_all["cc"]["target"].apply(lambda x: x / 10**6) 
    fps, pblatency = get_fps_pblatency(df_all["timing"])
    network_latency = roq.get_packet_latency(df_all["rtp"])
    network_latency = network_latency.set_axis(df_all["rtp"]["ts_s"])
    network_loss, sender_loss = losses(df_all["rtp"])

    fps = fps[~fps.index.duplicated(keep='first')]
    pblatency = pblatency[~pblatency.index.duplicated(keep='first')]
    tput_s = tput_s[~tput_s.index.duplicated(keep='first')]
    tput_p = tput_p[~tput_p.index.duplicated(keep='first')]
    network_latency = network_latency[~network_latency.index.duplicated(keep='first')]
    df_all["flights"] = df_all["flights"][~df_all["flights"]["ts"].duplicated(keep='first')]

    df_all["tput_s"] = tput_s
    df_all["tput_p"] = tput_p
    df_all["fps"] = fps
    df_all["pblatency"] = pblatency
    df_all["network_latency"] = network_latency
    df_all["network_loss"] = network_loss
    df_all["sender_loss"] = sender_loss
    
    fig = plot_latency(df_all)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

if __name__ == '__main__':
    main()
