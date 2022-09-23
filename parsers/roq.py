#!/usr/bin/env python3

import pandas as pd
import sqlite3
import argparse
import json
import os
import sys

def create_table_index(conn):
    c = conn.cursor()
    c.execute(f'''
CREATE TABLE IF NOT EXISTS "index" (
    "id"    INTEGER,
    "day"   TEXT NOT NULL,
    "flight"    INTEGER NOT NULL,
    "trajectory"    TEXT,
    "notes" TEXT,
    "pi"    INTEGER NOT NULL,
    "provider"  TEXT NOT NULL,
    "direction"     TEXT NOT NULL,
    "recorded_on"   TEXT NOT NULL,
    "type"  TEXT NOT NULL,
    PRIMARY KEY("id" AUTOINCREMENT),
    UNIQUE("day","flight","pi","provider","direction","recorded_on","type")
)''')
    conn.commit()

def get_index_ids(conn, info):
    conditions = [f"{k}='{v}'" for k, v in info.items()]
    where = ' AND '.join(conditions)
    sql = f'SELECT id FROM "index" WHERE {where}'
    c = conn.cursor()
    c.execute(sql)
    return [v for (v,) in c.fetchall()]

def new_index_id(conn, info):
    c = conn.cursor()
    c.execute('''INSERT INTO 'index' ('day', 'flight', 'pi', 'provider', 'direction', 'recorded_on', 'type')
    VALUES (:day, :flight, :pi, :provider, :direction, :recorded_on, :type)''', info)
    return c.lastrowid

def get_or_create_index_id(conn, info):
    create_table_index(conn)
    index_ids = get_index_ids(conn, info)
    if len(index_ids) == 0:
        return new_index_id(conn, info)
    return index_ids[0]


def get_info_from_filename(log_path):
    basename = os.path.basename(log_path)
    filename, ext = os.path.splitext(basename)
    log_type = filename.split('_')[2]

    if log_type == 'ping' and ext == '.txt':
        # 2202U_01_roq_scream_4_P1_pi,cc
        parts = filename.split('_')
        if len(parts) < 7:
            print("Filename not properly formatted, expecting at least 7 parts")
            sys.exit(1)
        return log_type, { 'day': parts[0], 'flight': parts[1], 'type': parts[3],
                         'pi': parts[4], 'provider': parts[5], 'recorded_on': parts[6] }

    if log_type == 'iperf':
        # 2202R_09_iperf_udp_down_4_P1_pi.json
        parts = filename.split('_')
        if len(parts) < 8:
            print("Filename not properly formatted, expecting at least 8 parts")
            sys.exit(1)
        return log_type, { 'day': parts[0], 'flight': parts[1], 'type': parts[3],
                           'direction': parts[4], 'pi': parts[5], 'provider': parts[6],
                           'recorded_on': parts[7] }

    if log_type == 'roq' or log_type == 'ping': # really ugly, but there are _ping_*.latency.csv files
        # 2202R_01_roq_gcc_2_P1_pi,rtp
        # 2202R_01_roq_gcc_2_P1_pi,fps
        name, data_type = filename.split('.')
        parts = name.split('_')
        if len(parts) < 7:
            print("Filename not properly formatted, expecting at least 7 parts")
            sys.exit(1)
        if data_type == 'cc':
            data_type = parts[3]
        return data_type, { 'day': parts[0], 'flight': parts[1], 'type': parts[3],
                            'pi': parts[4], 'provider': parts[5], 'recorded_on': parts[6] }
    else:
        print(f"Unknown log_type inferred from filename: {log_type}")
        sys.exit(1)

def get_table_for_data(data_type, protocol):
    if data_type == 'iperf':
        return f"{data_type}_{protocol}"
    if data_type in ['rtp', 'fps', 'gcc', 'scream', 'ping', 'timing', 'ssim', 'netem', 'emuho', 'rtpbuffer', 'latency', 'loss']:
        return data_type
    return None

def create_table(conn, table):
    c = conn.cursor()
    if table == 'rtp':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'seqnum_orig' INTEGER,
        'seqnum_unwr' INTEGER,
        'seqnum_cc' INTEGER,
        'marker' INTEGER,
        'size' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'fps':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'fps_cur' REAL,
        'fps_avg' REAL,
        'drop' REAL,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'gcc':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'target' INTEGER,
        'loss_target' INTEGER,
        'loss' INTEGER,
        'delay_target' INTERGER,
        'delay_meas' REAL,
        'delay_estim' REAL,
        'delay_thresh' REAL,
        'rtt' REAL,
        'usage' TEXT,
        'state' TEXT,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'scream':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'target' INTEGER,
        'queue_delay' REAL,
        'rtt' REAL,
        'cwnd' INTEGER,
        'bif' INTEGER,
        'lost' INTEGER,
        'rate_transm' INTEGER,
        'rate_acked' INTEGER,
        'hi_seq_acked' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'ping':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'rtt' REAL,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'iperf_udp':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts_start' TIMESTAMP,
        'ts_end' TIMESTAMP,
        'duration' REAL,
        'bytes' INTEGER,
        'bps' REAL,
        'packets' INTEGER,
        'packets_lost' INTEGER,
        'lost_percent' REAL,
        'jitter_ms' REAL,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts_start')
        )
        ''')
    elif table == 'iperf_tcp':
        print("iperf_tcp not yet implemented")
        sys.exit(1)
    # Parser for old format
    # elif table == 'timing':
    #     c.execute(f'''
    #     CREATE TABLE IF NOT EXISTS '{table}' (
    #     'index_id' INTEGER,
    #     'ts' TIMESTAMP,
    #     'frame_nr' INTEGER,
    #     'latency' INTEGER,
    #     'now' INTEGER,
    #     FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
    #     PRIMARY KEY('index_id','ts')
    #     )
    #     ''')
    elif table == 'timing':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'frame_nr' INTEGER,
        'latency' INTEGER,
        'time_s' INTEGER,
        'time_p' TEXT,
        'sec_offset' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'ssim':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'id' INTEGER,
        'frame_streamer' INTEGER,
        'frame_player' INTEGER,
        'ssim' REAL,
        'msg' TEXT,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','id')
        )
        ''')
    elif table == 'netem':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'rate' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'emuho':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'start_ts' TIMESTAMP,
        'end_ts' TIMESTAMP,
        'duration' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','start_ts')
        )
        ''')
    elif table == 'rtpbuffer':
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts' TIMESTAMP,
        'percent' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','ts')
        )
        ''')
    elif table == 'latency': # for importing csv files created with pcap_latency.py
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts_sent' TIMESTAMP,
        'ts_rcvd' TIMESTAMP,
        'seq' INTEGER,
        'latency_ms' REAL,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','seq')
        )
        ''')
    elif table == 'loss': # for importing csv files created with pcap_latency.py
        c.execute(f'''
        CREATE TABLE IF NOT EXISTS '{table}' (
        'index_id' INTEGER,
        'ts_sent' TIMESTAMP,
        'seq' INTEGER,
        FOREIGN KEY('index_id') REFERENCES 'index'('id') ON DELETE CASCADE ON UPDATE CASCADE,
        PRIMARY KEY('index_id','seq')
        )
        ''')
    else:
        print(f"Unknown table {table}")
        sys.exit(1)
    conn.commit()

def parse_ping(log_file, ping_type):
    tss = []
    rtts = []
    with open(log_file, 'r') as f:
        f.readline() # skip first line (header)
        for line in f:
            try:
                if ping_type == 'icmp' and 'DUP' in line:
                    print(f"duplicate pong received: {line}", end='')
                    continue
                ts, rest = line.split(';')
                parts = rest.split('=') # [... '51.5 ms']
                rtt, _ = parts[-1].split(' ')
                tss.append(ts)
                rtts.append(rtt)
            except Exception as e:
                print(f"Failed to parse line: {e}\n{line}")
    return pd.DataFrame.from_dict({ 'ts': pd.to_datetime(tss), 'rtt': rtts }).set_index('ts')

def parse_iperf(log_file, info):
    with open(log_file) as f:
        try:
            log = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to decode {log_file}: {e}")
            sys.exit(1)
    if not 'timestamp' in log['start']:
        print(f"Don't import {log_file}: no timestamp (transfer did not start?)")
        sys.exit(1)
    start_ts = pd.Timestamp(log['start']['timestamp']['timesecs'], unit='s', tz=0) # Time is in UTC

    sender = 'server' if info['direction'] == 'down' else 'pi'
    receiver = 'server' if info['direction'] == 'up' else 'pi'
    rows = []
    for interval in log['intervals'][:-1]: # Drop the last entry as that is the same as the penultimate entry
        data = interval['sum']
        start = start_ts + pd.Timedelta(data['start'], unit='s')
        end = start_ts + pd.Timedelta(data['end'], unit='s')
        if info['recorded_on'] == receiver:
            rows.append((start, end, data['seconds'], data['bytes'], data['bits_per_second'], data['jitter_ms'], data['lost_packets'], data['packets'], data['lost_percent']))
        if info['recorded_on'] == sender:
            rows.append((start, end, data['seconds'], data['bytes'], data['bits_per_second'], data['packets']))

    if info['recorded_on'] == receiver:
        columns = ['ts_start', 'ts_end', 'duration', 'bytes', 'bps', 'jitter_ms', 'packets_lost', 'packets', 'lost_percent']
    if info['recorded_on'] == sender:
        columns = ['ts_start', 'ts_end', 'duration', 'bytes', 'bps', 'packets']
    return pd.DataFrame.from_records(rows, columns=columns, index='ts_start')

def parse_data(table, log_file, info):
    if table == 'rtp':
        names = ['ts', 'pt', 'ssrc', 'seqnum_orig', 'timestamp', 'marker', 'size', 'seqnum_cc', 'seqnum_unwr']
        return pd.read_csv(log_file, sep='\t', index_col=0, parse_dates=True, names=names,
                           usecols=['ts', 'seqnum_orig', 'marker', 'size', 'seqnum_cc', 'seqnum_unwr'])
    elif table == 'fps':
        return pd.read_csv(log_file, sep='\t', index_col=0, parse_dates=True,
                           names=['ts', 'fps_cur', 'fps_avg', 'drop'])
    elif table == 'gcc':
        names = ['ts', 'target', 'loss_target', 'loss', 'delay_target', 'delay_meas', 'delay_estim', 'delay_thresh', 'rtt', 'usage', 'state']
        return pd.read_csv(log_file, sep='\t', index_col=0, parse_dates=True, names=names)
    elif table == 'scream':
        names = ['ts', 'target', 'queue_delay', 'rtt', 'cwnd', 'bif', 'lost', 'rate_transm', 'rate_acked', 'hi_seq_acked']
        df = pd.read_csv(log_file, sep='\t', index_col=0, parse_dates=True, names=names)
        df['rtt'] *= 1000 # convert from s to ms
        return df
    elif table == 'ping':
        return parse_ping(log_file, info['type'])
    elif table == 'iperf_udp':
        return parse_iperf(log_file, info)
    elif table == 'timing':
        df = pd.read_csv(log_file, sep='\t')
        df.drop(df.index[-1], inplace=True) # the last row is often not completly written to the file and causes problems
        if 'now' in df.columns: # old format: ts,frame_nr,latency,now
            df.rename(columns={'now': 'time_p'}, inplace=True)
        df = df.set_index(pd.to_datetime(df['ts'], utc=True)).drop('ts', axis='columns')
        for field in ['sec_offset','latency','time_p']:
            if field not in df:
                df[field] = -1
        df.loc[df['latency'] == -1, 'latency'] = -1000
        df.loc[:, 'latency'] /= 1000 # convert us to ms
        df['time_p'] = df['time_p'].apply(str) # time_p can be larger than 2**63-1 which is sqlite's INTEGER limit
        return df
    elif table == 'ssim':
        df = pd.read_csv(log_file, sep='\t')
        df.rename(columns={'ix_s': 'frame_streamer', 'ix_p': 'frame_player'}, inplace=True)
        df.index.name = 'id'
        return df
    elif table == 'netem':
        df = pd.read_csv(log_file, sep='\t')
        df = df.set_index(pd.to_datetime(df['ts'], utc=True)).drop('ts', axis='columns')
        return df
    elif table == 'emuho':
        df = pd.read_csv(log_file, sep='\t', names=['start_ts', 'duration'])
        df['start_ts'] = pd.to_datetime(df['start_ts'], utc=True)
        df['end_ts'] = df['start_ts'] + pd.to_timedelta(df['duration'], unit='ms')
        df.set_index('start_ts', inplace=True)
        return df
    elif table == 'rtpbuffer':
        df = pd.read_csv(log_file, sep='\t', names=['ts', 'percent'])
        df = df.set_index(pd.to_datetime(df['ts'], utc=True)).drop('ts', axis='columns')
        return df
    elif table == 'latency':
        df = pd.read_csv(log_file, sep='\t')
        df['ts_sent'] = pd.to_datetime(df['ts_sent'], utc=True)
        df['ts_rcvd'] = pd.to_datetime(df['ts_rcvd'], utc=True)
        df.rename(columns={'seq_unwr': 'seq'}, inplace=True)
        df.drop(columns='dst_port', inplace=True)
        df.drop_duplicates(subset=['seq'], keep='first', inplace=True, ignore_index=True)
        df.set_index('ts_sent', inplace=True)
        return df
    elif table == 'loss':
        df = pd.read_csv(log_file, sep='\t')
        df['ts_sent'] = pd.to_datetime(df['ts_sent'], utc=True)
        df.rename(columns={'seq_unwr': 'seq'}, inplace=True)
        df.drop(columns='dst_port', inplace=True)
        df.set_index('ts_sent', inplace=True)
        return df
    else:
        print(f"Unknown table {table}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description=f"Parse ROQ-test dump files into tables")
    parser.add_argument('log',  help='Dump file')
    parser.add_argument('database', help='Target sqlite database file')
    args = parser.parse_args()

    data_type, info = get_info_from_filename(args.log)
    if data_type == 'static':
        print("cc files of static transfers contain no data")
        return
    if data_type in ['rtp', 'fps', 'rtcp', 'gcc', 'scream', 'latency', 'loss', 'timing', 'ssim', 'netem', 'emuho', 'rtpbuffer', 'ping']:
        info['direction'] = 'up'

    conn = sqlite3.connect(args.database)
    index_id = get_or_create_index_id(conn, info)

    table = get_table_for_data(data_type, info['type'])
    if table is None:
        print(f"Unknown data type {data_type}")
        sys.exit(1)
    print(f"Insert {args.log} ({index_id=}) into table {table}")

    create_table(conn, table)
    df = parse_data(table, args.log, info)
    df['index_id'] = index_id
    try:
        df.set_index(df.index.tz_convert('utc'), inplace=True)
    except Exception as e:
        if df.index.name.startswith('ts') and not df.empty:
            print(e)
            breakpoint()
            sys.exit(1)
    df.replace('<nil>', 0, inplace=True)
    try:
        df.to_sql(table, conn, if_exists='append')
    except sqlite3.IntegrityError as e:
        print(f"Failed inserting data into {table}: {e}")

if __name__ == '__main__':
    main()
