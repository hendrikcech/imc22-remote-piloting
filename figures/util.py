#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

from matplotlib.image import imread
from tempfile import NamedTemporaryFile
def get_actual_figsize(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi, pad_inches=0)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

# Taken from https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
def set_actual_figsize(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_actual_figsize(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False

def where_clause_from_selector(s):
    conditions = []
    if 'flight' in s:
        conditions.append(f"flight = '{s['flight'].upper()}'")
    if 'recorded_on' in s:
        conditions.append(f"recorded_on = '{s['recorded_on'].lower()}'")
    if 'pi' in s:
        conditions.append(f"pi = '{s['pi']}'")
    if 'pis' in s:
        conditions.append(f"pi in {s['pis']}")
    if 'interface' in s:
        conditions.append(f"interface = '{s['interface']}'")
    if 'direction' in s:
        conditions.append(f"direction = '{s['direction'].lower()}'")
    if 'name' in s:
        conditions.append(f"name = '{s['name']}'")
    if 'provider' in s:
        conditions.append(f"interface = '{s['provider']}'")
    if 'protocol' in s:
        conditions.append(f"protocol = '{s['protocol']}'")
    if 'stream' in s:
        streams = s['stream']
        if not isinstance(s['stream'], list):
            streams = [s['stream']]
        stream_where = 'stream IN (' + ','.join(str(v) for v in streams) + ')'
        conditions.append(stream_where)
    where_clause = ' AND '.join(conditions)
    return where_clause

def get_test_data_from_table(conn, selector, table, columns=['*']):
    sql = f"SELECT {','.join(columns)} FROM {table}"
    where_clause = where_clause_from_selector(selector)
    if where_clause != '':
        sql += f" WHERE {where_clause}"
    sql += " ORDER BY ts ASC"
    df = pd.read_sql(sql, conn, parse_dates=['ts'], index_col='ts')
    if len(df) == 0:
        print(f"Query returned 0 rows: {sql}")
    return df

def check_streams_in_df(df, name):
    streams = df['stream'].value_counts()
    if len(streams) != 1:
        print(f"Note: Multiple streams in df '{name}':\n({streams})")

def get_handovers(conn, start, end, pi, modem):
    """Better use get_handovers_hit"""
    sql = f'''
    SELECT ts, ci FROM
        (SELECT LEAD(ci) OVER (PARTITION BY pi, modem ORDER BY ts) AS next_ci, *
        FROM modem
        WHERE ts BETWEEN '{str(start)}' AND '{str(end)}' AND pi = '{pi}' AND modem = '{modem}') t1
    WHERE t1.next_ci != t1.ci
    '''
    return pd.read_sql(sql, conn, index_col='ts', parse_dates=['ts'])


def get_handovers_hit(conn, start, end, pi, modem):
    sql = f'''
    SELECT start_ts, end_ts, duration, target_phys_cell, dl_freq, ul_freq, handover_type, t304 FROM
    handovers
    WHERE start_ts >= '{str(start)}' AND end_ts <= '{str(end)}' AND pi = '{pi}' AND provider = '{modem}'
    '''
    return pd.read_sql(sql, conn, parse_dates=['start_ts', 'end_ts'])

def table_exists(conn, table):
    return _struct_exists(conn, table, 'table')
def view_exists(conn, table):
    return _struct_exists(conn, table, 'view')
def _struct_exists(conn, table, sql_type):
    c = conn.cursor()
    c.execute(f"SELECT name FROM sqlite_master WHERE type='{sql_type}' AND name='{table}'")
    return len(c.fetchall()) == 1

def plot_handovers(ax, df_ho, **kwargs):
    """Better use get_handovers_hit and plot_handovers_hit"""
    ho = None
    for idx, row in df_ho.iterrows():
        # ho = ax.axvline(idx, ymin=ymin, ymax=ymax, label=label, color=color, alpha=alpha)
        ho = ax.axvline(idx, **kwargs)
    return ho

def plot_handovers_hit(ax, df_ho, **kwargs):
    ho = None
    for idx, row in df_ho.iterrows():
        ho = ax.axvspan(row['start_ts'], row['end_ts'], **kwargs)
    return ho

def at_rssi_to_dbm(v):
    if v == 99:
        return 0
    return -(113 - v*2)

def provider_pseudonym(provider):
    if provider.upper() == 'P1':
        return 'P1'
    if provider.upper() == 'P2':
        return 'P2'

def capitalize_cc_name(v):
    if v == 'scream':
        return 'SCReAM'
    if v == 'gcc':
        return 'GCC'
    if v == 'static':
        return 'Static'
    return v

# List of DataFrames
def describe_dfs(labels, dfs, save_path=None, stdout=True):
    rows = []
    decimals = 4
    for label, df in zip(labels, dfs):
        desc = df.describe().round(decimals)
        descT = pd.DataFrame(desc).transpose()
        # descT.insert(0, 'label', [label + ': ' + idx for idx in descT.index])
        descT.insert(2, 'median', df.median().round(decimals))
        if len(df.dtypes) > 1:
            descT.reset_index(inplace=True)
        descT.insert(0, 'label', label)
        rows.append(descT)
    df = rows[0].append(rows[1:], ignore_index=True)
    if save_path is not None:
       df.to_csv(save_path, index=False)
    if stdout:
        print(f"{save_path}:\n{df}")
    return df

def is_table_present(conn, table):
    try:
        c = conn.cursor()
        c.execute(f'SELECT * FROM {table}')
        return True
    except:
        return False

### RTP utilities (protocol level, tables gst_rtp_player/streamer)

rtp_seqnum_wrap_around_manual_fixes = {
    "FLY524-02-F2_rtp_tcp_up_2_P1": [('player', '2021-05-24 07:38:03+00:00')],
    "FLY524-01-F1_rtp_tcp_up_1_P2": [('player', '2021-05-24 07:26:15+00:00')],
    "FLY524-06-F5_rtp_tcp_up_2_P1": [('player', '2021-05-24 08:13:53+00:00')], # 6 != 5
    "FLY524-06-F5_rtp_udp_up_3_P2": [], # 6 != 2
}

def fix_rtp_seqnum_wrap_around(dfu, name):
    prev_values = { 'seqnum_streamer': 0, 'seqnum_player': 0 }
    n_wraps = { 'seqnum_streamer': 0, 'seqnum_player': 0 }
    manual_wrap_idxs = {}
    for (party, ts) in (rtp_seqnum_wrap_around_manual_fixes[name] if name in rtp_seqnum_wrap_around_manual_fixes else []):
        df = dfu[dfu[f'seqnum_{party}'].notnull()]
        idx = df[df['ts'] >= pd.to_datetime(ts)].iloc[0].name
        manual_wrap_idxs[idx] = True
    def update(idx, row, key):
        v = row[key]
        if np.isnan(v):
            return
        if prev_values[key] - v > 30000: # wrap-around
            n_wraps[key] += 1
        if idx in manual_wrap_idxs:
            print(f"Applying manual wrap at {idx}")
            n_wraps[key] += 1
        prev_values[key] = v
        dfu.at[idx, key] = v + 0xFFFF * n_wraps[key]
    for idx, row in dfu.iterrows():
        update(idx, row, 'seqnum_streamer')
        update(idx, row, 'seqnum_player')
    if n_wraps['seqnum_streamer'] != n_wraps['seqnum_player']:
        print(f"Number of wrap-arounds don't match: streamer={n_wraps['seqnum_streamer']} != player={n_wraps['seqnum_player']:}")

def merge_rtp_streamer_player_dfs(dfs, dfp):
    dfu = pd.merge(dfs, dfp, how='outer', left_index=True, right_index=True, indicator=True,
                   suffixes=('_streamer', '_player'))
    dfu.sort_index(inplace=True)
    dfu['ts'] = dfu.index
    dfu.set_index(dfu.index - dfu.index[0], inplace=True) # change to relative index (datetime -> timedelta)
    return dfu

def get_rtp_dfs_by_name(conn, name_prefix):
    dfs = get_test_data_from_table(conn, { 'name': name_prefix + '_pi' }, 'gst_rtp_streamer')
    dfp = get_test_data_from_table(conn, { 'name': name_prefix + '_server' }, 'gst_rtp_player')
    if dfs.empty or dfp.empty:
        return None
    return merge_rtp_streamer_player_dfs(dfs, dfp)

def get_rtp_ts_diff(df):
    view = df[df['_merge'] == 'left_only']
    diff_ts = view['timestamp_streamer'] - view['timestamp_player']
    return diff_ts[diff_ts < 2e6]

# Taken from https://stackoverflow.com/a/11886564
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        # points = points[:,None]
        points = points.to_numpy()[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


### Matplotlib helpers

def timedelta_mmss_formatter_func(x, pos):
    minutes = int((x / 1e+9) // 60)
    seconds = int((x / 1e+9) % 60)
    # return f"{minutes:02d}:{seconds:02d}"
    total_seconds = int(x / 1e+9)
    return str(total_seconds)
timedelta_mmss_formatter = mticker.FuncFormatter(timedelta_mmss_formatter_func)

def use_style():
    plot_style = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_style.txt")
    plt.style.use(plot_style)

def plot_cdf(ax, data, **kwargs):
    if False:
        return ax.hist(data, cumulative=True, density=1, bins=1000, histtype='step', label=label)
    else:
        x = np.sort(data)
        fx = np.array(range(len(data))) / float(len(data))
        return ax.plot(x, fx, **kwargs)

def plot_external_legend(parts, figsize, **kwargs): # ncol =
    fig_legend = plt.figure(figsize=figsize)
    fig_legend.legend(parts, [p.get_label() for p in parts], **kwargs)
    fig_legend.tight_layout()
    return fig_legend


# --- ROQ utils
def in_clause_values(values):
    return ",".join(f"'{v}'" for v in values)

def ids_with(conn, key, value):
    if type(value) is list:
        where = f'`{key}` IN ({in_clause_values(value)})'
    else:
        where = f"`{key}`='{value}'"
    c = conn.cursor()
    c.execute("SELECT id FROM `index` WHERE "+where)
    return set(v for v, in c.fetchall())

def get_air_ts(conn, index_id_streamer, air_thresh=0):
    c = conn.cursor()
    c.execute(f'''WITH
rtp_start AS (SELECT ts FROM rtp WHERE index_id='{index_id_streamer}' ORDER BY ts ASC LIMIT 1),
rtp_end AS (SELECT ts FROM rtp WHERE index_id='{index_id_streamer}' ORDER BY ts DESC LIMIT 1),
air_start AS (SELECT flight.ts FROM flight, rtp_start, rtp_end WHERE flight.ts BETWEEN rtp_start.ts AND rtp_end.ts AND relative_height > {air_thresh} ORDER BY flight.ts ASC LIMIT 1),
air_end AS (SELECT flight.ts FROM flight, rtp_start, rtp_end WHERE flight.ts BETWEEN rtp_start.ts AND rtp_end.ts AND relative_height > {air_thresh} ORDER BY flight.ts DESC LIMIT 1)
SELECT * FROM air_start UNION ALL SELECT * FROM air_end
''')
    rows = c.fetchall()
    if len(rows) == 0:
        return pd.NaT, pd.NaT # no in-air time during this test
    return pd.to_datetime(rows[0][0]), pd.to_datetime(rows[1][0])

def get_air_filter(conn, idxs, ts_col):
    filter = None
    for start, end in [get_air_ts(conn, idx, air_thresh=10) for idx in idxs]:
        exp = (ts_col >= start) & (ts_col <= end)
        if filter is None:
            filter = exp
        else:
            filter |= exp
    if filter is None:
        return [False] * len(ts_col)
    return filter



# To use in figure's figsize
mm = (1 / 2.54) / 10
textwidth = 5.8091 # in inch
textwidth_springer = 4.8211
columnwidth_acmart = 3.35 # inch
fulltextwidth_acmart = 7.5 # inch
textwidth_acmart = 7 # inch
threshold_color = 'crimson' # red

def mpl_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


### Test information

def intersect_flight_and_names(flights, names):
    names_flight = [name[:12] for name in names]
    return set(flights + names_flight)

rural_udp_down_lim = ['FLY526-01-F1_iperf_udp_down_3_P2', 'FLY526-01-F1_iperf_udp_down_4_P1', 'FLY526-04-F2_iperf_udp_down_1_P2',
                      'FLY526-04-F2_iperf_udp_down_2_P1', 'FLY526-05-F8_iperf_udp_down_1_P2', 'FLY526-05-F8_iperf_udp_down_2_P1',
                      'FLY526-05-F8_iperf_udp_down_3_P2', 'FLY526-05-F8_iperf_udp_down_4_P1', 'FLY526-10-F1_iperf_udp_down_4_P1',
                      'FLY526-10-F1_iperf_udp_down_3_P2', 'FLY526-14-F1_iperf_udp_down_4_P1', 'FLY526-14-F1_iperf_udp_down_3_P2',
                      'FLY526-15-F8_iperf_udp_down_1_P2', 'FLY526-15-F8_iperf_udp_down_3_P2', 'FLY526-15-F8_iperf_udp_down_2_P1',
                      'FLY526-15-F8_iperf_udp_down_4_P1']
rural_udp_down_unlim = ['FLY526-02-F6_iperf_udp_down_4_P1', 'FLY526-02-F6_iperf_udp_down_3_P2', 'FLY526-06-F3_iperf_udp_down_3_P2',
                        'FLY526-06-F3_iperf_udp_down_4_P1', 'FLY526-11-F7_iperf_udp_down_3_P2',
                        'FLY526-11-F7_iperf_udp_down_4_P1', 'FLY526-12-F3_iperf_udp_down_4_P1', 'FLY526-12-F3_iperf_udp_down_3_P2',
                        'FLY526-13-F7_iperf_udp_down_4_P1', 'FLY526-13-F7_iperf_udp_down_3_P2', 'FLY526-16-F6_iperf_udp_down_4_P1',
                        'FLY526-16-F6_iperf_udp_down_3_P2']
rural_udp_up_unlim = [ 'FLY526-11-F7_iperf_udp_up_3_P2', 'FLY526-11-F7_iperf_udp_up_4_P1', 'FLY526-06-F3_iperf_udp_up_3_P2',
                       'FLY526-06-F3_iperf_udp_up_4_P1', 'FLY526-12-F3_iperf_udp_up_4_P1', 'FLY526-12-F3_iperf_udp_up_3_P2',
                       'FLY526-13-F7_iperf_udp_up_4_P1', 'FLY526-13-F7_iperf_udp_up_3_P2']

rural_tcp_down_lim = ['FLY526-01-F1_iperf_tcp_down_1_P2', 'FLY526-01-F1_iperf_tcp_down_2_P1', 'FLY526-04-F2_iperf_tcp_down_3_P2',
                      'FLY526-04-F2_iperf_tcp_down_4_P1', 'FLY526-10-F1_iperf_tcp_down_1_P2', 'FLY526-10-F1_iperf_tcp_down_2_P1',
                      'FLY526-14-F1_iperf_tcp_down_1_P2', 'FLY526-14-F1_iperf_tcp_down_2_P1']
rural_tcp_down_unlim = ['FLY526-02-F6_iperf_tcp_down_1_P2', 'FLY526-02-F6_iperf_tcp_down_2_P1', 'FLY526-16-F6_iperf_tcp_down_1_P2',
                        'FLY526-16-F6_iperf_tcp_down_2_P1']
rural_tcp_up_unlim = ['FLY526-03-F4_iperf_tcp_up_1_P2', 'FLY526-03-F4_iperf_tcp_up_2_P1', 'FLY526-05-F8_iperf_tcp_up_3_P2',
                        'FLY526-05-F8_iperf_tcp_up_4_P1', 'FLY526-06-F3_iperf_tcp_up_1_P2', 'FLY526-06-F3_iperf_tcp_up_2_P1',
                        'FLY526-08-F4_iperf_tcp_up_1_P2', 'FLY526-08-F4_iperf_tcp_up_2_P1', 'FLY526-11-F7_iperf_tcp_up_1_P2',
                        'FLY526-12-F3_iperf_tcp_up_1_P2', 'FLY526-12-F3_iperf_tcp_up_2_P1', 'FLY526-13-F7_iperf_tcp_up_1_P2',
                        'FLY526-15-F8_iperf_tcp_up_3_P2', 'FLY526-15-F8_iperf_tcp_up_4_P1']

urban_udp_down_lim = [
'FLY524-01-F1_iperf_udp_down_3_P2', 'FLY524-01-F1_iperf_udp_down_4_P1', 'FLY524-02-F2_iperf_udp_down_1_P2',
    'FLY524-02-F2_iperf_udp_down_2_P1', 'FLY524-08-F8_iperf_udp_down_1_P2', 'FLY524-08-F8_iperf_udp_down_2_P1',
    'FLY524-08-F8_iperf_udp_down_3_P2', 'FLY524-08-F8_iperf_udp_down_4_P1', 'FLY524-10-F1_iperf_udp_down_3_P2',
    'FLY524-10-F1_iperf_udp_down_4_P1', 'FLY524-13-F2_iperf_udp_down_1_P2', 'FLY524-13-F2_iperf_udp_down_2_P1',
    'FLY524-14-F8_iperf_udp_down_1_P2', 'FLY524-14-F8_iperf_udp_down_2_P1', 'FLY524-14-F8_iperf_udp_down_3_P2',
    'FLY524-14-F8_iperf_udp_down_4_P1']

urban_udp_down_unlim = [
    'FLY524-03-F3_iperf_udp_down_3_P2', 'FLY524-03-F3_iperf_udp_down_4_P1', 'FLY524-04-F3_iperf_udp_down_3_P2',
    'FLY524-04-F3_iperf_udp_down_4_P1', 'FLY524-07-F6_iperf_udp_down_3_P2', 'FLY524-07-F6_iperf_udp_down_4_P1',
    'FLY524-11-F7_iperf_udp_down_3_P2', 'FLY524-11-F7_iperf_udp_down_4_P1', 'FLY524-12-F7_iperf_udp_down_3_P2',
    'FLY524-12-F7_iperf_udp_down_4_P1', 'FLY524-15-F6_iperf_udp_down_3_P2', 'FLY524-15-F6_iperf_udp_down_4_P1']

urban_udp_up_unlim = [
    'FLY524-03-F3_iperf_udp_up_3_P2', 'FLY524-03-F3_iperf_udp_up_4_P1', 'FLY524-04-F3_iperf_udp_up_3_P2',
    'FLY524-04-F3_iperf_udp_up_4_P1', 'FLY524-11-F7_iperf_udp_up_3_P2', 'FLY524-11-F7_iperf_udp_up_4_P1',
    'FLY524-12-F7_iperf_udp_up_3_P2', 'FLY524-12-F7_iperf_udp_up_4_P1']



urban_tcp_down_lim = [
    'FLY524-01-F1_iperf_tcp_down_1_P2', 'FLY524-01-F1_iperf_tcp_down_2_P1', 'FLY524-02-F2_iperf_tcp_down_3_P2',
    'FLY524-02-F2_iperf_tcp_down_4_P1', 'FLY524-10-F1_iperf_tcp_down_1_P2', 'FLY524-10-F1_iperf_tcp_down_2_P1',
    'FLY524-11-F7_iperf_tcp_up_1_P2', 'FLY524-12-F7_iperf_tcp_up_1_P2' ]

urban_tcp_down_unlim = [
    'FLY524-03-F3_iperf_tcp_up_1_P2', 'FLY524-03-F3_iperf_tcp_up_2_P1', 'FLY524-04-F3_iperf_tcp_up_1_P2',
    'FLY524-04-F3_iperf_tcp_up_2_P1', 'FLY524-04-F4_iperf_tcp_up_1_P2', 'FLY524-04-F4_iperf_tcp_up_2_P1',
    'FLY524-15-F6_iperf_tcp_down_1_P2', 'FLY524-15-F6_iperf_tcp_down_2_P1']

urban_tcp_up_unlim = [
    'FLY524-07-F6_iperf_tcp_down_1_P2', 'FLY524-07-F6_iperf_tcp_down_2_P1', 'FLY524-08-F8_iperf_tcp_up_3_P2',
    'FLY524-08-F8_iperf_tcp_up_4_P1', 'FLY524-11-F7_iperf_tcp_up_1_P2', 'FLY524-12-F7_iperf_tcp_up_1_P2',
    'FLY524-14-F8_iperf_tcp_up_3_P2', 'FLY524-14-F8_iperf_tcp_up_4_P1' ]


urban_grd_udp_up_lim = [ 'GRD108-01-G1_iperf_udp_up_1_P2',
                         'GRD108-03-G1_iperf_udp_up_1_P2', 'GRD108-05-G1_iperf_udp_up_1_P1',
                         'GRD108-01-G1_iperf_udp_up_2_P2', 'GRD108-03-G1_iperf_udp_up_2_P2',
                         'GRD108-05-G1_iperf_udp_up_2_P1']
urban_grd_udp_down_lim = [ 'GRD108-07-G2_iperf_udp_down_1_P2',
                           'GRD108-07-G2_iperf_udp_down_2_P2']
urban_grd_udp_up_unlim = [ 'GRD108-02-G3_iperf_udp_up_1_P1',
                           'GRD108-04-G3_iperf_udp_up_1_P2', 'GRD108-06-G3_iperf_udp_up_1_P1',
                           'GRD108-02-G3_iperf_udp_up_2_P1', 'GRD108-04-G3_iperf_udp_up_2_P2',
                           'GRD108-06-G3_iperf_udp_up_2_P1', 'GRD108-10-G3_iperf_udp_up_1_P2',
                           'GRD108-10-G3_iperf_udp_up_2_P2' ]
urban_grd_udp_down_unlim = [ 'GRD108-08-G4_iperf_udp_down_1_P2',
                             'GRD108-08-G4_iperf_udp_down_2_P2', 'GRD108-09-G4_iperf_udp_down_1_P1',
                             'GRD108-09-G4_iperf_udp_down_2_P1']
