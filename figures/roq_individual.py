#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import sqlite3
import pandas as pd
import util
# import config_matplotlibrc
import numpy as np

base_ts = None
end_ts = None
def rel_index(tss):
    base_ts = tss[0]
    try:
        try:
            return (tss - base_ts).total_seconds()
        except:
            return (tss - base_ts).dt.total_seconds()
    except Exception as e:
        if len(tss) == 0:
            return tss
        raise e

def rel_indexx(tss, base_ts):
    try:
        try:
            return (tss - base_ts).total_seconds()
        except:
            return (tss - base_ts).dt.total_seconds()
    except Exception as e:
        if len(tss) == 0:
            return tss
        raise e

def get_throughput(df):
    tput = df.groupby('index_id').resample("1s")['size'].sum() * 8 / 1e6
    return tput.reset_index(level=0, drop=True).sort_index()
def rtp_throughput(ax, df_s, df_p):
    throughput_s = get_throughput(df_s.set_index('ts'))
    throughput_p = get_throughput(df_p.set_index('ts'))
    p1 = ax.plot(rel_index(throughput_s.index), throughput_s, label='streamer')[0]
    p2 = ax.plot(rel_index(throughput_p.index), throughput_p, label='player')[0]
    ax.legend(ncol=3)
    ax.set_ylim(bottom=-1)

def handovers(ax, df, base_ts):
    p = None
    for _, row in df.iterrows():
        p = ax.axvspan(xmin=rel_indexx(row['start_ts'], base_ts), xmax=rel_indexx(row['end_ts'], base_ts), color='dimgrey', alpha=0.6, zorder=4)
    return p

def handovers_ci(ax, df):
    x = []
    y = []
    start_times = rel_index(df['start_ts'])
    for i in range(len(df)-1):
        if df.iloc[i+1]['end_ts'] < base_ts:
            # Skip all handovers until we are at the last handover that happend
            # just before the window of time that we are looking at.
           continue
        if df.iloc[i]['start_ts'] < base_ts:
            # This is the last handover that happend before our time window.
            y.append(df.iloc[i]['target_phys_cell'])
            y.append(df.iloc[i]['target_phys_cell'])
            x.append(rel_index(base_ts))
            x.append(start_times.iloc[i+1])
        else:
            y.append(df.iloc[i]['target_phys_cell'])
            y.append(df.iloc[i]['target_phys_cell'])
            x.append(start_times.iloc[i])
            if df.iloc[i+1]['start_ts'] > end_ts:
                # We plotted the HOs until the end of the time window
                x.append(rel_index(end_ts))
                break
            else:
                x.append(start_times.iloc[i+1])
        # print(df.iloc[i]['target_phys_cell'])
    ax.plot(x, y, label='Cell ID', color='purple')
    ax.set_ylabel('Cell ID')
    cis = set(y)
    ax.yaxis.set_major_locator(mticker.FixedLocator(list(cis)))

def netem(ax, df_netem):
    points = []
    last_rate = None
    for ts, rate in  df_netem['rate'].iteritems():
        if last_rate is not None:
            points.append((ts, last_rate))
        points.append((ts, rate))
        last_rate = rate
    ax.plot([(e[0] - base_ts).total_seconds() for e in points], [e[1] for e in points], label='netem rate')
    ax.legend(ncol=4)
    ax.set_ylim(bottom=0)

def get_packet_latency(df):
    return (df['ts_p'] - df['ts_s']).dt.total_seconds() * 1000
def packet_latency(ax, df):
    latency = get_packet_latency(df)
    ax.plot(rel_index(df['ts_s']), latency, label='Packet latency')

def get_playback_latency(df_timing):
    filter = [
        df_timing['latency'] != -1,
        df_timing['frame_nr'] >= df_timing['frame_nr'].shift(),
    ]
    df = df_timing[df_timing['latency'] != -1]
    df = df.reset_index()
    while True:
        df['frame_nr_max'] = df['frame_nr'].expanding().max()
        df = df[df['frame_nr'] >= df['frame_nr_max']]
        df['ts_diff'] = (df['ts'] - df['ts'].shift()).dt.total_seconds()*1000 # ms
        df['ts_diff'].fillna(0, inplace=True)
        df['latency_diff'] = df['latency'] - df['latency'].shift()
        df['latency_diff'].fillna(0, inplace=True)
        df_len = len(df)
        remove = df[df['latency_diff'] > df['ts_diff'] + 100]
        df = df[df['latency_diff'] <= df['ts_diff'] + 100]
        if len(df) == df_len:
            break
        # print(f"Pruned {df_len - len(df)}/{len(df)} invalied entries from latency.")
        # print("Removed: {remove}")
    df = df.set_index('ts')
    return df['latency']
def playback_latency(ax, df_timing):
    df = get_playback_latency(df_timing)
    ax.plot(rel_index(df.index), df, label='Playback latency')

def get_ift(df_timing, series=True): # intra-frame time
    if len(df_timing) == 0:
        return pd.Series()
    df = df_timing.reset_index().sort_values(by=['index_id', 'ts'])
    df['ift'] = df.groupby('index_id')['ts'].diff().dt.total_seconds() * 1000
    if series:
        return df['ift'].dropna()
    else:
        return df
    return ift_ms

# def get_stalls(df_timing):
def stalls(ax, df_timing):
    df = df_timing.reset_index()
    df['ift'] = df.groupby('index_id')['ts'].diff()
    stalls = df[df['ift'] > pd.Timedelta('300ms')]
    for ts, ift in zip(stalls['ts'], stalls['ift']):
        p = ax.axvspan(xmin=rel_index(ts), xmax=rel_index(ts + ift), zorder=0, color='orange')
        print(f"Stall from {rel_index(ts)} to {rel_index(ts + ift)}")

def get_timing_based_fps(df_timing, reset_index=True):
    freq = 500
    # fps_df = df_timing.groupby(pd.Grouper(level='ts', freq=f"{freq}ms")).count() * (1000/freq)
    # return fps_df['index_id'] # return an arbitrary row
    fps_df = df_timing.groupby('index_id').resample(f"{freq}ms").count()['index_id'] * (1000/freq)
    if reset_index:
        return fps_df.reset_index(level=0, drop=True).sort_index() # index=(index_id, ts) -> ts
    else:
        fps_df.name = 'fps'
        return fps_df # contains multiindex
def timing_based_fps(ax, df_timing):
    fps = get_timing_based_fps(df_timing)
    try:
        ax.plot(rel_index(fps.index), fps, label='FPS')
    except:
        breakpoint()
    ax.set_ylabel('fps')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(15))

def frame_delta_time(ax, df_timing):
    df = df_timing.reset_index().set_index('ts', drop=False)['ts'].diff().dt.total_seconds() * 1000
    df_r = df.resample('100ms').mean().dropna()
    ax.plot(rel_index(df_r.index), df_r, label='inter frame time (ms)')
    ax.axhline(1000/30, label='ideal inter frame time')
    ax.set_ylim(bottom=0, top=df_r[df_r < 1000].max())

def fps(ax, df_fps):
    fps = df_fps['fps_cur'].rolling('500ms').mean()
    ax.plot(rel_index(fps.index), fps, label='FPS')
    ax.axhline(30, alpha=0.4, zorder=0)

def get_network_losses(df):
    return df[pd.isna(df['size_p'])]
def get_sender_losses(df):
    prev_ts = 0
    prev_seqnum = df["seqnum_unwr"][0]
    losses_ts = []
    losses_seqnum = []
    for ts_s, seqnum_unwr in zip(df['ts_s'], df['seqnum_unwr']):
        if seqnum_unwr != prev_seqnum:
            for lost_seqnum in np.arange(prev_seqnum+1, seqnum_unwr):
                losses_ts.append(prev_ts + (ts_s - prev_ts)/2)
                losses_seqnum.append(lost_seqnum)
        prev_ts = ts_s
        prev_seqnum = seqnum_unwr
    return pd.DataFrame(data={'ts_s': losses_ts, 'seqnum_unwr': losses_seqnum})

def fmt_stem(stem):
    markerline, stemlines, baseline, = stem
    markerline.set_markersize(2)
    markerline.set_markerfacecolor(None)
    stemlines.set_linewidth(1)
    baseline.set_visible(False)

def losses(ax_net, df):
    # Don't look at losses in the last 10 s of
    # the transmission. If the player is stopped before the streamer, false
    # positives are reported.
    cutoff = (df['ts_s'].iloc[-1]) - pd.Timedelta('10s')
    df_lens = df[df['ts_s'] < cutoff]
    df_net_loss = get_network_losses(df_lens)
    df_snd_loss = get_sender_losses(df_lens)
    all_sec = df_lens.resample('1s', on='ts_s').count()['index_id_s']
    label_net = 'Loss percentage (network)'
    label_snd = 'Loss percentage (sender)'
    if not df_net_loss.empty:
        lost_sec = df_net_loss.resample('1s', on='ts_s').count()['ts_s']
        lost_perc = (lost_sec / all_sec).fillna(0)
        lost_perc = lost_perc[lost_perc > 0]
        # ax_net.plot(rel_index(lost_perc.index), lost_perc * 100, label=label_net)
        stem = ax_net.stem(rel_index(lost_perc.index), lost_perc * 100, label=label_net,
                           markerfmt='C0.', use_line_collection=True)
        fmt_stem(stem)
    ax_snd = ax_net.twinx()
    if not df_snd_loss.empty:
        lost_sec = df_snd_loss.resample('1s', on='ts_s').count()['ts_s']
        lost_sec = lost_sec[lost_sec > 0]
        stem = ax_snd.stem(rel_index(lost_sec.index), lost_sec, label=label_snd,
                           linefmt='C1-', markerfmt='C1.', use_line_collection=True)
        fmt_stem(stem)
        ax_snd.set_ylabel('Dropped by\nSCReAM')
        ax_snd.spines['right'].set_visible(util.mpl_colors()[1])
        ax_snd.spines['right'].set_edgecolor(util.mpl_colors()[1])
        ax_snd.tick_params(color=util.mpl_colors()[1])

    ax_net.set_ylim(bottom=0)
    ax_net.set_ylabel('Loss %\nsndr/netw')
    ax_net.spines['left'].set_color(util.mpl_colors()[0])
    ax_net.tick_params(color=util.mpl_colors()[0])
    ax_snd.set_ylim(bottom=0)

def rtpbuffer(ax, df):
    perc = df[df['percent'] > 0]['percent']
    stem = ax.stem(rel_index(perc.index), perc, label='RTP Buffer % filled',
                        markerfmt='C0.', use_line_collection=True)
    fmt_stem(stem)
    ax.set_ylabel('RTP buffer (%)')
    ax.set_ylim(bottom=0, top=max(10, max(perc)))

def altitude(ax, df):
    ax.plot(rel_index(df.index), df['relative_height'])
    ax.set_ylabel('Altitude (m)')
    ax.set_ylim(bottom=-3, top=130)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(40))

def velocity(ax, df):
    ax.plot(rel_index(df.index), df['vel_comp'])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_ylim(bottom=-1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

def ssim(ax, df):
    ax.plot(rel_index(df['ts']), df['ssim'], label='SSIM')
    ax.set_ylim(bottom=0, top=1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(.25))

def rssi(ax, df):
    ax.plot(rel_index(df.index), df[df['rssi'] != -1]['rssi'], label='RSSI')
    ax.set_ylabel('RSSI')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))

def save_highlight_plots(ax, df, info, save_path):
    '''
    ax is used to control the xlim parameters
    df contains time points that should be plotted. relevant key is ts.
    save_path is path to folder into which the plots are saved
    '''
    export_intervals = []
    window_size = pd.Timedelta('5s')
    while len(df) > 0:
        # print(f"len(df)={len(df)}")
        start = df.iloc[0]['ts']
        end = start + window_size
        while True:
            expand = df[(df['ts'] > end) & (df['ts'] < (end + window_size))]
            # print(f"len(expand)={len(expand)}, start={start}, end={end}")
            if len(expand) == 0 or (end - start) > window_size*2:
                break
            end = expand.iloc[-1]['ts'] + window_size
        # print(f"Add ({start}, {end}) to export_intervals")
        export_intervals.append((start, end))
        df = df[df['ts'] > end]

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    os.makedirs(save_path, exist_ok=True)
    print(f"Save {len(export_intervals)} highlight plots")
    for start, end in export_intervals:
        left = rel_index(start-window_size)
        right = rel_index(end)
        ax.set_xlim(left=left, right=right)
        filename = get_save_name(info) + f"-{int(left)}-{int(right)}"
        text = ax.get_figure().text(0, 0, f"{start-window_size}-{end}", fontsize='small')
        # print(f"Save {left}-{right} to {filename}")
        plt.savefig(f"{os.path.join(save_path, filename)}.png", bbox_inches='tight')
        text.set_visible(False)


def get_info(conn, index_id_streamer):
    c = conn.cursor()
    cols = ['day', 'flight', 'trajectory', 'notes', 'pi', 'provider', 'type', 'recorded_on']
    c.execute(f"SELECT {','.join(cols)} FROM `index` WHERE id='{index_id_streamer}'")
    info_row = c.fetchone()
    if info_row == None:
        print(f"Did not find index id {index_id_streamer}")
        return
    cols[6] = 'rtp_cc' # rename type -> rtp_cc
    info = { k: v for k,v in zip(cols, info_row) }
    if info['recorded_on'] == 'server' or info['rtp_cc'] not in ['gcc', 'scream', 'static']:
        print(f"Pass streamer index id of RTP transfer")
        return

    c.execute("SELECT id FROM `index` WHERE day=? AND flight=? AND pi=? AND recorded_on='server' AND type=?",
              (info['day'], info['flight'], info['pi'], info['rtp_cc']))
    result = c.fetchone()
    if result is None:
        print(f"No player id for {index_id_streamer} found")
        return
    index_id_player = result[0]
    print(f"Match with player id {index_id_player}")

    info['air_ts'] = util.get_air_ts(conn, index_id_streamer)
    info['index_id_streamer'] = index_id_streamer
    info['index_id_player'] = index_id_player
    return info

def get_save_name(info):
    trajectory_s = [info['trajectory']] if info['trajectory'] is not None else []
    flight_f = f"{info['flight']:02}" if type(info['flight']) == int else info['flight']
    notes_f = [info['notes'].replace(' ', "").replace('/', '_').replace('.', '_')] if info['notes'] is not None else []
    return '-'.join([info['day'], flight_f] + trajectory_s + notes_f + [info['provider'], info['rtp_cc']])

def get_suptitle(info):
    if info['day'].endswith('U'):
        location = 'urban'
    elif info['day'].endswith('R'):
        location = 'rural'
    elif info['day'].endswith('H'):
        location = 'home'
    else:
        location = info['day']
    trajectory_s = [info['trajectory']] if info['trajectory'] is not None else []
    notes_s = [info['notes']] if info['notes'] is not None else []
    suptitle = [info['day'], location, f"flight {info['flight']} ({info['index_id_streamer']})"] + trajectory_s  + notes_s  + [f"Pi #{info['pi']}", info['provider'], info['rtp_cc']]
    return ", ".join(suptitle)

def main():
    global base_ts, end_ts
    parser = argparse.ArgumentParser(description="Plot individual ROQ run")
    parser.add_argument('database', help='The sqlite database file')
    parser.add_argument('index_id_streamer', help='Select what to plot')
    parser.add_argument('--save', help='Save figure to this folder')
    parser.add_argument('--save-ssim-drops', help='Save SSIM drops to this folder')
    parser.add_argument('--save-latency-spikes', help='Save latency spikes to this folder')
    parser.add_argument('--save-handovers', help='Save handovers to this folder')
    # parser.add_argument('index_id_player', help='Select what to plot')
    args = parser.parse_args()

    util.use_style()

    conn = sqlite3.connect(args.database)

    info = get_info(conn, args.index_id_streamer)
    if info is None:
        return

    df_s = pd.read_sql(f"SELECT * FROM rtp WHERE index_id='{info['index_id_streamer']}' ORDER BY ts", conn, parse_dates=['ts'])
    df_p = pd.read_sql(f"SELECT * FROM rtp WHERE index_id='{info['index_id_player']}' ORDER BY ts", conn, parse_dates=['ts'])
    df = pd.merge(df_s, df_p, how='left', on='seqnum_unwr', suffixes=('_s', '_p'))
    base_ts = df_s.loc[0, 'ts']
    end_ts = max(df.iloc[-1]['ts_s'], df.iloc[-1]['ts_p'])

    df_timing = pd.read_sql(f"SELECT * FROM timing WHERE index_id='{info['index_id_player']}' ORDER BY ts",
                            conn, parse_dates=['ts'], index_col='ts')

    df_cc = None
    if info['rtp_cc'] != 'static':
        df_cc = pd.read_sql(f"SELECT * FROM `{info['rtp_cc']}` WHERE index_id='{args.index_id_streamer}' ORDER BY ts", conn, parse_dates=['ts'], index_col='ts')

    df_tssim = pd.read_sql(f"SELECT id, frame_streamer, frame_player, ssim FROM ssim WHERE index_id='{info['index_id_player']}' ORDER BY id", conn)
    df_netem = pd.read_sql(f"SELECT * FROM netem WHERE index_id='{info['index_id_streamer']}' ORDER BY ts", conn, parse_dates=['ts'], index_col='ts')
    df_flight = pd.read_sql(f"SELECT ts, relative_height, vel_comp FROM flight WHERE ts>='{base_ts}' AND ts<='{end_ts}'ORDER BY ts",
                            conn, parse_dates=['ts'], index_col='ts')
    df_modem = pd.read_sql(f"SELECT ts, rssi, ci FROM modem WHERE ts>='{base_ts}' AND ts<='{end_ts}' AND pi='{info['pi']}' ORDER BY ts",
                            conn, parse_dates=['ts'], index_col='ts')
    df_ho = pd.read_sql(f"SELECT * FROM handovers WHERE pi='{info['pi']}' ORDER BY start_ts",
                        conn, parse_dates=['start_ts', 'end_ts'])
    df_emuho = pd.read_sql(f"SELECT * FROM emuho WHERE index_id='{info['index_id_streamer']}' ORDER BY start_ts",
                        conn, parse_dates=['start_ts', 'end_ts'])
    df_rtpbuf = pd.read_sql(f"SELECT * FROM rtpbuffer WHERE index_id='{info['index_id_player']}' ORDER BY ts",
                                conn, parse_dates=['ts'], index_col='ts')

    rows = ['fps', 'ssim', 'tput', 'lat', 'loss']
    gridspec = [1, 1, 2, 2, 1]
    if not df_rtpbuf.empty: # RSSI
        rows.append('rtpbuf')
        gridspec += [1]
    if not df_flight.empty: # altitude, velocity
        rows += ['alt', 'vel']
        gridspec += [1, 1]
    if not df_modem.empty: # RSSI
        rows.append('rssi')
        gridspec += [1]
    if not df_ho.empty: # cell ID
        rows.append('cid')
        gridspec += [1]
    figsize = (util.textwidth_springer, util.textwidth_springer*1.6)
    fig, axes_i = plt.subplots(nrows=len(rows), figsize=figsize, sharex=True, gridspec_kw={'height_ratios': gridspec})
    fig.suptitle(get_suptitle(info))
    axes = { label: axes_i[idx] for idx, label in enumerate(rows) }

    timing_based_fps(axes['fps'], df_timing)
    stalls(axes['fps'], df_timing)
    # df_fps = pd.read_sql(f"SELECT * FROM fps WHERE index_id='{index_id_player}'", conn, parse_dates=['ts'], index_col='ts')
    # fps(axes['fps'], df_fps)

    df_ssim = None
    if len(df_tssim) > 0:
        key = 'frame_streamer'
        if False:
            df_tssim = df_tssim[df_tssim[key] != 0]
            df_ssim = pd.merge(df_timing.reset_index(), df_tssim, left_on='frame_nr', right_on=key, how='right')
            df_ssim['ts'].fillna(method='ffill', inplace=True)
            df_ssim = df_ssim[['ts', 'ssim']]
        else:
            df_ssim = df_tssim
            df_ssim['ts'] = base_ts + pd.to_timedelta(df_ssim[key] / 30, unit='s')
        ssim(axes['ssim'], df_ssim)
    axes['ssim'].set_ylabel('SSIM')

    rtp_throughput(axes['tput'], df_s, df_p)
    axes['tput'].set_ylabel('Mbps')
    if df_cc is not None:
        axes['tput'].plot(rel_index(df_cc.index), df_cc['target'] / 1e6, label='CC target')

    if len(df_netem) > 0:
        netem(axes['tput'], df_netem)
    # else:
    #     axes['tput'].axhline(5, label='Target bitrate (Mbps)')

    packet_latency(axes['lat'], df)
    playback_latency(axes['lat'], df_timing)
    if max(get_playback_latency(df_timing).max(), get_packet_latency(df).max()) >= 1000:
        axes['lat'].set_yscale("symlog", linthresh=1000)
        axes['lat'].yaxis.set_major_formatter(mticker.ScalarFormatter())
        minor_locators = list(np.arange(100, 1000, 100)) + list(np.arange(1000, 10000, 1000))
        minor_locators = [v for v in minor_locators if v <= axes['lat'].get_ylim()[1]]
        axes['lat'].yaxis.set_minor_locator(mticker.FixedLocator(minor_locators))
        # axes['lat'].yaxis.set_minor_formatter(mticker.ScalarFormatter())
        # axes['lat'].yaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10, linthresh=1000))
        axes['lat'].ticklabel_format(useOffset=False, style='plain')
    axes['lat'].set_ylabel('Latency (ms)')
    axes['lat'].legend(ncol=2)
    axes['lat'].set_ylim(bottom=0)
    # frame_delta_time(axes['lat'], df_timing)

    losses(axes['loss'], df)

    if 'rtpbuf' in axes:
        rtpbuffer(axes['rtpbuf'], df_rtpbuf)
    if 'alt' in axes:
        altitude(axes['alt'], df_flight)
    if 'vel' in axes:
        velocity(axes['vel'], df_flight)
    if 'rssi' in axes:
        rssi(axes['rssi'], df_modem)
    if 'cid' in axes:
        df_ho['start_ts'] = df_ho['start_ts'].dt.tz_localize('UTC')
        df_ho['end_ts'] = df_ho['end_ts'].dt.tz_localize('UTC')
        df_ho_lens = df_ho[(df_ho['start_ts'] >= base_ts) & (df_ho['end_ts'] <= end_ts)]
        handovers_ci(axes['cid'], df_ho)
        for ax in [ax for label, ax in axes.items() if label != 'cid']:
            handovers(ax, df_ho_lens)

    if not df_emuho.empty:
        df_emuho['start_ts'] += pd.Timedelta('1h')
        df_emuho['end_ts'] += pd.Timedelta('1h')
        for ax in axes.values():
            handovers(ax, df_emuho)

    axes_i[-1].set_xlabel('Time (s)')
    fig.tight_layout()

    if args.save_ssim_drops:
        ssim_change = df_ssim['ssim'] - df_ssim.shift(30)['ssim']
        drops = df_ssim[ssim_change < -0.25]
        save_highlight_plots(axes_i[-1], drops, info, args.save_ssim_drops)

    if args.save_latency_spikes:
        latency = get_playback_latency(df_timing)
        latency_diff = latency.rolling('1s').apply(lambda values: values[-1] - values[0], raw=True)
        latency_diff.fillna(0, inplace=True)
        spikes = latency[(latency_diff > latency*.3) & (latency_diff >= 100)].reset_index()
        save_highlight_plots(axes_i[-1], spikes, info, args.save_latency_spikes)

    if args.save_handovers:
        df_ho_lens = df_ho[(df_ho['start_ts'] >= base_ts) & (df_ho['end_ts'] <= end_ts)]
        ho_plot = df_ho_lens.rename(columns={'start_ts': 'ts'})
        save_highlight_plots(axes_i[-1], ho_plot, info, args.save_handovers)


    axes_i[-1].set_xlim(left=-10, right=rel_index(end_ts)+10)
    axes_i[-1].xaxis.set_major_locator(mticker.MultipleLocator(60))
    axes_i[-1].xaxis.set_minor_locator(mticker.MultipleLocator(10))



    if args.save:
        os.makedirs(args.save, exist_ok=True)
        filename = get_save_name(info)
        plt.savefig(f"{os.path.join(args.save, filename)}.png", bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()
