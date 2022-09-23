#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import sqlite3
import datetime
import dateutil.parser
import pandas as pd
import os
import sys

dummy_date = '1970-01-01T00:00:00Z'
def combine_gps_offset_time(offset, gps):
    if not (isinstance(offset, float) and (isinstance(gps, str) or isinstance(gps, float))):
        raise Exception("pandas, try next mode")
    if pd.isna(gps):
        gps = dummy_date
    date = dateutil.parser.isoparse(gps)
    seconds = datetime.timedelta(seconds=offset)
    return date + seconds

def dont_combine(offset, gps):
    if pd.isna(gps):
        gps = dummy_date
    return dateutil.parser.isoparse(gps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse UAV log file.')
    parser.add_argument('log',  help='UAV flight log file')
    parser.add_argument('database', help='Target sqlite database file')
    # Flights in 2021 May: flight_start
    parser.add_argument('--guess-ms')
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.log))[0]

    print(f"Importing {name}", end='')

    if args.guess_ms is not None:
        date_parser = combine_gps_offset_time
    else:
        date_parser = dont_combine
    df = pd.read_csv(args.log,
                    dtype={ 'offsetTime': float },
                    usecols=[47, 1, 38, 2, 3, 41, 19, 20, 21, 22, 34, 48],
                    parse_dates={'ts': [0, 10]}, keep_date_col=True,
                    date_parser=date_parser)
    df.drop(columns='GPS:dateTimeStamp', inplace=True)

    rename_dict = {
        'offsetTime':'offset_time',
        'IMU_ATTI(0):Longitude': 'lng',
        'IMU_ATTI(0):Latitude': 'lat',
        'IMU_ATTI(0):velN': 'vel_n',
        'IMU_ATTI(0):velE': 'vel_e',
        'IMU_ATTI(0):velD': 'vel_d',
        'IMU_ATTI(0):velComposite': 'vel_comp',
        'IMU_ATTI(0):distanceTravelled': 'distance_travelled',
        'flightTime': 'flight_time',
        'General:relativeHeight': 'relative_height',
        'GPS(0):heightMSL': 'height_msl'
    }
    df.rename(columns=rename_dict, inplace=True)

    df.insert(1, "flight", [name] * len(df))

    # Filter out incomplete records from start-up phase
    first_valid_date = datetime.datetime(year=2021, month=1, day=1, tzinfo=datetime.timezone.utc)
    df = df[df['ts'] > first_valid_date]

    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df.set_index('ts', inplace=True)

    # df_re = df.resample('100ms').mean()

    conn = sqlite3.connect(args.database)

    c = conn.cursor()
    c.execute('''
CREATE TABLE IF NOT EXISTS 'flight' (
    'ts' TIMESTAMP,
    'flight' TEXT,
    'offset_time' REAL,
    'lng' REAL,
    'lat' REAL,
    'vel_n' REAL,
    'vel_e' REAL,
    'vel_d' REAL,
    'vel_comp' REAL,
    'distance_travelled' REAL,
    'flight_time' REAL,
    'relative_height' REAL,
    'height_msl' REAL,
    PRIMARY KEY('ts', 'offset_time', 'flight'))''')
    conn.commit()

    df.to_sql('flight', conn, if_exists='append')

    print(": success!")
