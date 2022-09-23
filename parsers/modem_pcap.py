#!/usr/bin/env python3

import argparse
import sqlite3
import subprocess
from io import StringIO
import pandas as pd
import sys
import math

def pcap_to_csv(pcap):
    tshark = subprocess.run([
        "tshark",
        "-r", args.pcap,
        "-C", "ISODateTime",
        "-T", "fields",
        "-e", "_ws.col.UTC_ISO",
        "-e", "frame.number",
        "-e", "frame.time_relative",
        "-e", "frame.time_delta_displayed",
        "-e", "lte-rrc.c1",
        "-e", "lte-rrc.rrc_TransactionIdentifier",
        "-e", "lte-rrc.targetPhysCellId",
        "-e", "lte-rrc.dl_CarrierFreq",
        "-e", "lte-rrc.ul_CarrierFreq",
        "-e", "lte-rrc.dl_Bandwidth",
        "-e", "lte-rrc.ul_Bandwidth",
        "-e", "lte-rrc.handoverType",
        "-e", "lte-rrc.t304",
        "-e", "lte-rrc.rsrpResult",
        "-e", "lte-rrc.rsrqResult",
        "-e", "lte-rrc.physCellId",
        "-E", "header=y",
        "-E", "separator=;", "-E", "quote=n",
        "-E", "aggregator=/s",
        "-Y", "(lte-rrc.c1 == 1 and lte-rrc.criticalExtensions == 0) or lte-rrc.c1 == 2 or (lte-rrc.c1 == 4 and lte-rrc.targetPhysCellId)",
    ], capture_output=True)
    print(f"tshark finished ({tshark.returncode})")
    if tshark.returncode != 0:
        print(tshark.stderr)
        sys.exit(1)
    return tshark.stdout.decode('unicode_escape')

handover_columns = ['start_ts', 'end_ts', 'duration', 'target_phys_cell',
                    'dl_freq', 'ul_freq', 'dl_bw', 'ul_bw', 'handover_type', 't304']
def get_handover_row(reconf, reconf_comp):
    return [reconf.name, reconf_comp.name,
           reconf_comp['frame.time_delta_displayed'],
           int(reconf['lte-rrc.targetPhysCellId']),
           int(reconf['lte-rrc.dl_CarrierFreq']),
           int(reconf['lte-rrc.ul_CarrierFreq']) if not math.isnan(reconf['lte-rrc.ul_CarrierFreq']) else -1,
           int(reconf['lte-rrc.dl_Bandwidth']) if not math.isnan(reconf['lte-rrc.dl_Bandwidth']) else -1,
           int(reconf['lte-rrc.ul_Bandwidth']) if not math.isnan(reconf['lte-rrc.ul_Bandwidth']) else -1,
           int(reconf['lte-rrc.handoverType']),
           reconf['lte-rrc.t304']]

def find_handovers(df):
    cur_reconf = None
    last_meas_cells = ''
    ho_rows = []
    for index, row in df.iterrows():
        if '4' in str(row['lte-rrc.c1']): # Sometimes this is a string, sometimes a float
            if cur_reconf is None: # Reconfiguration start
                cur_reconf = index
            else:
                print(f"Encountered another reconfiguration message while waiting for complete message for {cur_reconf}.")
                sys.exit(1)
            if str(int(row['lte-rrc.targetPhysCellId'])) not in last_meas_cells:
                print(f"Handover initiated at {index} but targetPhysCellId {int(row['lte-rrc.targetPhysCellId'])} is not included in last measurement report ({last_meas_cells})")
        if str(row['lte-rrc.c1']) == '2' and cur_reconf is not None: # Reconfiguration complete
            tid = df.loc[cur_reconf, 'lte-rrc.rrc_TransactionIdentifier']
            if tid == row['lte-rrc.rrc_TransactionIdentifier']:
                ho_row = get_handover_row(df.loc[cur_reconf], row)
                ho_rows.append(ho_row)
                cur_reconf = None
            else:
                print("Complete message immediately following reconfiguration message is about another transaction")
        if '1' in str(row['lte-rrc.c1']) and not pd.isna(row['lte-rrc.physCellId']): # Measurement Report
            last_meas_cells = str(row['lte-rrc.physCellId'])
    return pd.DataFrame(ho_rows, columns=handover_columns)

def insert_handovers(df_ho, db_path, pi, provider):
    conn = sqlite3.connect(db_path)

    c = conn.cursor()
    table_name = 'handovers'
    c.execute(f'''
CREATE TABLE IF NOT EXISTS '{table_name}' (
    'pi' INTEGER,
    'provider' TEXT,
    'start_ts' TEXT,
    'end_ts' TEXT,
    'duration' FLOAT,
    'target_phys_cell' INTEGER,
    'dl_freq' INTEGER,
    'ul_freq' INTEGER,
    'dl_bw' INTEGER,
    'ul_bw' INTEGER,
    'handover_type' INTEGER,
    't304' INTEGER,
    PRIMARY KEY('pi', 'provider', 'start_ts'))''')
    conn.commit()

    df_ho['pi'] = pi
    df_ho['provider'] = provider

    df_ho.to_sql(table_name, conn, if_exists='append', index=False)

    print(f"Appended {len(df_ho)} rows to table '{table_name}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and import modem pcap.')
    parser.add_argument('pi',  help='Pi id', type=int)
    parser.add_argument('provider',  help='Provider', choices=['P1', 'P2'])
    parser.add_argument('pcap',  help='Modem pcap')
    parser.add_argument('database', help='Target sqlite database file')
    args = parser.parse_args()

    csv = pcap_to_csv(args.pcap)
    df = pd.read_csv(StringIO(csv), sep=';', parse_dates=[0], index_col=0)
    df.index.rename('ts', inplace=True)
    df_ho = find_handovers(df)
    insert_handovers(df_ho, args.database, args.pi, args.provider)
