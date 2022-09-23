#!/usr/bin/env python3

import argparse
import datetime
import sqlite3
import sys
import traceback
import dateutil.parser
from collections import Counter

def get_mcc_mnc(file):
    with open(file) as f:
        mccs = []
        mncs = []
        i = -1
        try:
            for i, line in enumerate(f.readlines()):
                if '+CEN1' in line:
                    _, mcc, mnc = [int(v) for v in line[32+7:].strip().split(',')]
                    mccs.append(mcc)
                    mncs.append(mnc)
        except Exception as e:
            print(f"Error in line {i}:\n{e}")

        mccs_cntr = Counter(mccs)
        mncs_cntr = Counter(mncs)

        if len(mccs_cntr) != 1 or len(mncs_cntr) != 1:
            print(f"Multiple MCCs or MNCs in input file: {mccs_cntr}, {mncs_cntr}.")
        if len(mccs_cntr) == 0 or len(mncs_cntr) == 0:
            sys.exit(1)

        mcc = mccs_cntr.most_common(1)[0][0]
        mnc = mncs_cntr.most_common(1)[0][0]

        return mcc, mnc

def empty_row():
    return {
        'time': None,
        'rssi': None,
        'ber': None,
        'lac': None,
        'ci': None,
        'act': None
    }

def conv_int(v, base=10):
    try:
        return int(v, base=base)
    except:
        return -1

def parse_file(file):
    rows = []
    with open(file) as f:
        row = empty_row()

        try:
            for i, line in enumerate(f.readlines()):
                row['time'] = line[1:30] + 'Z'
                if '+CSQ:' in line:
                    values = [v for v in line[32+6:].strip().split(',')]
                    if len(values) < 2:
                        print(f"Not rssi and ber found in: {line}")
                        continue
                    row['rssi'] = conv_int(values[0])
                    row['ber'] = conv_int(values[1])
                elif '+CREG:' in line:
                    parts = line[32+7:].strip().split(',')
                    def get(i):
                        if i >= len(parts):
                            return '-1'
                        return parts[i]
                    if get(0) == '2':
                        row['lac'] = get(2).strip('"')
                        row['ci'] = get(3).strip('"')
                        row['act'] = conv_int(get(4).strip(), base=16)
                    elif get(0) == '1':
                        row['lac'] = get(1).strip('"')
                        row['ci'] = get(2).strip('"')
                        row['act'] = conv_int(get(3).strip(), base=16)
                    elif not get(0) == '0':
                        print(f"unexpected CREG: {line}")
                        # sys.exit(1)
                # TODO: parse QCRSRP
                if all(v is not None for v in row.values()): # row complete
                    rows.append(row)
                    row = empty_row()
        except:
            print(f"Error while parsing line {i+1}: {line}")
            print(parts)
            traceback.print_exc()
            sys.exit(1)
    return rows

def trim_file_to_timerange(rows, start, end):
    return [row for row in rows
            if row['time'] >= start and row['time'] <= end]

def fix_timezone(dt):
    if dt.tzinfo is None:
        # default_tz = datetime.timezone(datetime.timedelta(hours=1))
        return dt.replace(tzinfo=datetime.timezone.utc)
    else:
        return dt

def match_mnc_to_carrier(mnc):
    m = {1: 'P1', 3: 'P2'}
    return m[mnc]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse output of modem (AT commands).')
    parser.add_argument('database', help='The sqlite database file')
    parser.add_argument('pi', type=int, help='The pi id')
    parser.add_argument('file', help='The log file')
    parser.add_argument('--start', default='1970-01-01 00:00:00Z', help='Import info from this time on (ISO format)')
    parser.add_argument('--end',  default='9970-01-01 00:00:00Z', help='Import info up to this time (ISO format)')
    args = parser.parse_args()

    start = fix_timezone(dateutil.parser.parse(args.start))
    end = fix_timezone(dateutil.parser.parse(args.end))

    conn = sqlite3.connect(args.database, 100000)

    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS modem
        (ts TEXT,
        pi INTEGER,
        modem TEXT,
        mnc INTEGER,
        rssi INTEGER,
        ber INTEGER,
        lac TEXT,
        ci TEXT,
        act INTEGER,
        PRIMARY KEY (ts, pi, modem))''')
    conn.commit()

    mcc, mnc = get_mcc_mnc(args.file)

    rows = parse_file(args.file)
    print(f"Import mcc={mcc}, mnc={mnc} from {start} to {end}: {len(rows)} rows")
    # for row in rows:
    #     row['time'] = fix_timezone(row['time'])
    # rows = trim_file_to_timerange(rows, start, end)

    c = conn.cursor()
    modem = match_mnc_to_carrier(mnc)
    c.executemany(f'''
    INSERT OR REPLACE INTO modem
        (ts, pi, modem, mnc, rssi, ber, lac, ci, act)
        VALUES (?, "{args.pi}", "{modem}", "{mnc}",  ?, ?, ?, ?, ?)''',
                [list(row.values()) for row in rows])
    conn.commit()
