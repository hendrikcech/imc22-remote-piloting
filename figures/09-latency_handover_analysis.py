import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import roq_individual as roq
import util
pd.options.mode.chained_assignment = None


def main():
    parser = argparse.ArgumentParser(description="Analyze latency around handovers")
    parser.add_argument("database", help="Database")
    parser.add_argument("--save", help="Save plots to this file")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    util.use_style()

    hos, sql_index = fetch_data_from_sql(conn)
    index_ids_streamer = sql_index["id"]
    network_latencies_all = fetch_latencies(conn, index_ids_streamer)
    network_latencies_all.index = pd.to_datetime(network_latencies_all.index)

    collect_latency_stats_all = []
    collect_latency_ho_all, collect_latency_bfho_all, collect_latency_afho_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 

    # iterate over each ho instance, define time windows around hos,
    # and collect latency stats
    ho_prev = pd.DataFrame()
    index_drop = []
    for index, row in hos.iterrows():
        if not ho_prev.empty:
            time_diff = abs((row["start_ts"] - ho_prev["end_ts"]).total_seconds())
            if time_diff < 5 and row["pi"] == ho_prev["pi"] and row["provider"] == ho_prev["provider"]:
                print("============= DROP HO =============")
                print(f"index: {index}, timestamp: {row['start_ts']}")
                print(f"index: {index-1}, timestamp: {ho_prev['end_ts']}\n")
                index_drop.append(index-1)
                index_drop.append(index)
        ho_prev = row
    
    index_drop = set(index_drop) # remove duplicates
    hos.drop(index_drop, inplace=True)
    for i, (index, row) in enumerate(hos.iterrows()):
        # find potential index ids for the ho occurence
        print("\n============ NEW LOOP ============")
        print(f"row['provider']:\n{row['provider']}")
        print(f"row['pi']:\n{row['pi']}")
        print(f"row['start_ts']:\n{row['start_ts']}")
        index_ids = sql_index[(sql_index["provider"] == row["provider"]) & (sql_index["pi"] == row["pi"])]["id"]
        if len(index_ids) == 0:
            print(f"No matching index data could be found for HO index: {index+1}\n")
            continue
        
        # fetch all latency data of the index ids, where ho could be occured
        network_latencies = network_latencies_all[(network_latencies_all["index_id"].isin(index_ids))]
        network_latencies = network_latencies.sort_index()
        network_latencies = network_latencies[~network_latencies.index.duplicated(keep='first')]
        collect_latency_stats = []

        if not network_latencies.empty:
            # find out where ho occured in the timing table
            index_matched_time = network_latencies.index.get_indexer([row["start_ts"]], "nearest")[0]
            print(f"ho timestamp: {row['start_ts']}")
            print(f"matched timestamp at flight: {network_latencies.index[index_matched_time]}")
            print(f"time diff: {(row['start_ts']-network_latencies.index[index_matched_time]).total_seconds()}")
            if abs((row['start_ts']-network_latencies.index[index_matched_time]).total_seconds()) > 1: 
                print(f"time difference between matched and actual ho time is too large."
                    f"{abs((row['start_ts']-network_latencies.index[index_matched_time]).total_seconds())} s."
                    "Skipping this handover")
                continue
            network_latencies_per_flight = network_latencies[(network_latencies["index_id"] == network_latencies["index_id"].iloc[index_matched_time])]
            # find out the lower/upper index numbers (not 'index_id') in the timing table
            # to create time windows before, after and during hos
            index_time_windows = get_index_time_windows(row, network_latencies_per_flight)
            index_ho_upper = index_time_windows[0]
            
            # crop the time windows of before/after/during hos from timing table 
            time_windows = get_time_windows(network_latencies_per_flight, index_time_windows)
            time_window_ho = time_windows[0]     
            time_window_beforeho = time_windows[1]
            time_window_afterho = time_windows[2]
        
            # compute latency stats
            minmax_latencies = get_minmax_latency(network_latencies["latency"], index_ho_upper, 
                                        time_window_ho, time_window_beforeho, time_window_afterho)
            latency_ratios = compute_latency_ratios(network_latencies["latency"], index_ho_upper, minmax_latencies)
                
            # merge all latency data, latency stats and related info from corresponding ho event under a single list
            collect_latency_ho_all, collect_latency_bfho_all, collect_latency_afho_all, collect_latency_stats = collect_latencies(
                                                                    network_latencies_per_flight["latency"], collect_latency_ho_all, 
                                                                    collect_latency_bfho_all, collect_latency_afho_all, 
                                                                    collect_latency_stats, index_ho_upper, time_window_ho, 
                                                                    time_window_beforeho, time_window_afterho, latency_ratios)

        print(f"collect latency stats: {collect_latency_stats}")
        collect_latency_stats_all.append(collect_latency_stats)
        print(f"completed {i}/{len(hos)}")


    # convert collected latency info to a pd dataframe
    latency_stats_info = create_latency_table(collect_latency_ho_all, collect_latency_bfho_all, 
                                                            collect_latency_afho_all, collect_latency_stats_all)

    # extract relevant info from big latency table to generate plot
    latency_ratio, latency_min_max = extract_latencies(latency_stats_info)
    latency_ratio, latency_min_max = stack_latencies(latency_ratio, latency_min_max)
    latency_ratio.rename(columns = {'latency_ratio_ho':'Around Handovers', 
                            'latency_ratio_bfho':'Before Handovers', 
                            'latency_ratio_afho': 'After Handovers'}, 
                            inplace = True)
    
    ######### plot the ratios
    util.use_style()

    figsize = (util.columnwidth_acmart, 1)
    fig1, ax1 = plt.subplots(figsize=figsize)
    my_dict = {'After HO': latency_ratio["latency_ratio"][(latency_ratio["time_window"] == "latency_ratio_afho")],
                'Before HO': latency_ratio["latency_ratio"][(latency_ratio["time_window"] == "latency_ratio_ho")]}
 
    bp = ax1.boxplot(my_dict["After HO"], positions=[0], 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.015, 
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))
    plt.plot([], c=util.mpl_colors()[0], label="After HO")

    for attr in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[attr], color='black')
    
    for patch in bp['boxes']:
        patch.set(facecolor=util.mpl_colors()[0])  
    
    bp2 = ax1.boxplot(my_dict["Before HO"], positions=[0.025], 
        showmeans=True, showfliers=True, vert=False, patch_artist=True, widths=0.015, 
        flierprops=dict(marker='.'),
        meanprops=dict(markerfacecolor=util.mpl_colors()[5], 
        markeredgecolor=util.mpl_colors()[5]))
    plt.plot([], c=util.mpl_colors()[1], label="Before HO")

    for attr in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp2[attr], color='black')
    
    for patch in bp2['boxes']:
        patch.set(facecolor=util.mpl_colors()[1])  

    ax1.set_yticklabels(my_dict.keys())     
    
    plt.grid(axis='y')

    plt.legend(frameon=False)
    plt.xlim([0, 40])
    plt.yticks([0, 0.025], ['', ''])
    plt.ylim([-0.0175, 0.0495])
    plt.ylabel("")
    plt.xlabel("Latency Ratio (Max./Min.)")
    plt.xscale("linear")
    util.set_actual_figsize(fig1, figsize)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        plt.savefig(args.save, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

def fetch_data_from_sql(conn):
    hos = pd.read_sql(f"SELECT provider, pi, start_ts, end_ts FROM handovers ORDER BY start_ts", # SELECT * FROM handovers WHERE start_ts >= '' and end_ts <= ''
                      conn, parse_dates=['start_ts', 'end_ts'])

    sql_index = pd.read_sql(f"SELECT provider, pi, id FROM 'index' WHERE recorded_on = 'pi'", conn)
    return hos,sql_index

def stack_latencies(latency_ratio, latency_min_max):
    latency_ratio = latency_ratio.stack().reset_index()
    latency_ratio.columns = ["index", "time_window", "latency_ratio"]
    latency_min_max = None
    return latency_ratio, latency_min_max

def extract_latencies(latency_stats_info):
    latency_ratio = latency_stats_info[['latency_ratio_ho', 'latency_ratio_bfho', 'latency_ratio_afho']].copy()
    latency_min_max = None
    return latency_ratio,latency_min_max

def create_latency_table(collect_latency_ho_all, collect_latency_bfho_all, 
                        collect_latency_afho_all, collect_latency_stats_all):
    latency_stats_info = pd.DataFrame(collect_latency_stats_all, 
                                columns=[
                                "latency_ratio_ho", "latency_ratio_bfho", "latency_ratio_afho"])

    collect_latency_ho_all.columns = ["latency"]
    collect_latency_bfho_all.columns = ["latency"]
    collect_latency_afho_all.columns = ["latency"]
    
    collect_latency_ho_all["time_window"] = len(collect_latency_ho_all) * ["Around\nHandover"]
    collect_latency_bfho_all["time_window"] = len(collect_latency_bfho_all) * ["Before\nHandover"]
    collect_latency_afho_all["time_window"] = len(collect_latency_afho_all) * ["After\nHandover"]
                     
    return latency_stats_info

def collect_latencies(network_latencies, collect_latency_ho_all, 
                    collect_latency_bfho_all, collect_latency_afho_all, 
                    collect_latency_stats, index_ho_upper, time_window_ho, 
                    time_window_beforeho, time_window_afterho, latency_ratios):
    
    if index_ho_upper != len(network_latencies.index) - 1:
        latencies_ho = network_latencies[time_window_ho.loc[(time_window_ho == True)].index]
        latencies_ho = latencies_ho.dropna()
        latencies_bfho = network_latencies[time_window_beforeho.loc[(time_window_beforeho == True)].index]
        latencies_bfho = latencies_bfho.dropna()
        latencies_afho = network_latencies[time_window_afterho.loc[(time_window_afterho == True)].index]
        latencies_afho = latencies_afho.dropna()
        if not latencies_ho.empty and not latencies_bfho.empty and not latencies_afho.empty:
            collect_latency_afho_all = pd.concat([collect_latency_afho_all, latencies_afho])
            collect_latency_afho_all = collect_latency_afho_all[~collect_latency_afho_all.index.duplicated(keep='first')]
            collect_latency_ho_all = pd.concat([collect_latency_ho_all, latencies_ho])
            collect_latency_ho_all = collect_latency_ho_all[~collect_latency_ho_all.index.duplicated(keep='first')]
            collect_latency_bfho_all = pd.concat([collect_latency_bfho_all, latencies_bfho])
            collect_latency_bfho_all = collect_latency_bfho_all[~collect_latency_bfho_all.index.duplicated(keep='first')]
        else:
            pass

    collect_latency_stats = append_multi(collect_latency_stats, latency_ratios)    
    collect_latency_ho_all = collect_latency_ho_all.dropna()
    collect_latency_bfho_all = collect_latency_bfho_all.dropna()
    collect_latency_afho_all = collect_latency_afho_all.dropna()

    return collect_latency_ho_all, collect_latency_bfho_all, collect_latency_afho_all, collect_latency_stats

def compute_latency_ratios(network_latencies, index_ho_upper, minmax_latencies):
    ratio_ho = minmax_latencies[1]/ minmax_latencies[0]
    ratio_beforeho = minmax_latencies[3]/minmax_latencies[2]
    
    if index_ho_upper != len(network_latencies.index) - 1:
        ratio_afterho = minmax_latencies[5]/minmax_latencies[4]
    else:
        ratio_afterho = np.nan
    
    print(f"latency ratio before ho: {ratio_beforeho}")
    print(f"latency ratio ho: {ratio_ho}")
    print(f"latency ratio after ho: {ratio_afterho}")
    latency_ratios = [ratio_ho, ratio_beforeho, ratio_afterho]
                        
    return latency_ratios

def get_minmax_latency(network_latencies, index_ho_upper, time_window_ho, 
                    time_window_beforeho, time_window_afterho):
    min_latency_afterho = None
    max_latency_afterho = None

    min_latency_ho = network_latencies[time_window_ho.loc[(time_window_ho == True)].index][(network_latencies > -1)].min()
    max_latency_ho = network_latencies[time_window_ho.loc[(time_window_ho == True)].index].max()
    min_latency_beforeho = network_latencies[time_window_beforeho.loc[(time_window_beforeho == True)].index][(network_latencies > -1)].min()
    max_latency_beforeho = network_latencies[time_window_beforeho.loc[(time_window_beforeho == True)].index].max()
    if index_ho_upper != len(network_latencies.index) - 1:
        min_latency_afterho = network_latencies[time_window_afterho.loc[(time_window_afterho == True)].index][(network_latencies > -1)].min()
        max_latency_afterho = network_latencies[time_window_afterho.loc[(time_window_afterho == True)].index].max()

    return min_latency_ho,max_latency_ho,min_latency_beforeho,max_latency_beforeho,min_latency_afterho,max_latency_afterho

def get_time_windows(network_latencies, index_time_windows):
    index_ho_upper = index_time_windows[0]
    index_ho_lower = index_time_windows[1]
    index_beforeho_upper = index_time_windows[2]
    index_beforeho_lower =  index_time_windows[3]
    index_afterho_upper = index_time_windows[4]
    index_afterho_lower = index_time_windows[5]
    time_window_afterho = None

    time_window_ho = network_latencies.index.to_series().between(network_latencies.index[index_ho_lower], network_latencies.index[index_ho_upper])
    time_window_beforeho = network_latencies.index.to_series().between(network_latencies.index[index_beforeho_lower], network_latencies.index[index_beforeho_upper])
    if index_ho_upper != len(network_latencies.index) - 1:
        time_window_afterho = network_latencies.index.to_series().between(network_latencies.index[index_afterho_lower], network_latencies.index[index_afterho_upper])
    return time_window_ho,time_window_beforeho,time_window_afterho

def get_index_time_windows(row, network_latencies):
    index_afterho_lower, index_afterho_upper = None, None
    index_ho_upper = network_latencies.index.get_indexer([row["start_ts"] + timedelta(seconds=0)], "nearest")[0]
    index_ho_lower = network_latencies.index.get_indexer([row["start_ts"] - timedelta(seconds=1.00)], "nearest")[0]
    index_beforeho_upper = index_ho_lower - 1
    index_beforeho_lower = network_latencies.index.get_indexer([row["start_ts"] - timedelta(seconds=2)], "nearest")[0]
    if index_ho_upper == len(network_latencies.index) - 1:
        print("after ho period cannot be calculated for last row of timing table")
    else:
        index_afterho_upper = network_latencies.index.get_indexer([row["start_ts"] + timedelta(seconds=1)], "nearest")[0]
        index_afterho_lower = index_ho_upper + 1
    return index_ho_upper,index_ho_lower,index_beforeho_upper,index_beforeho_lower,index_afterho_upper,index_afterho_lower

def fetch_latencies(conn, index_ids_streamer):
    column_names = ["index_id", "latency"]
    network_latencies_all, network_latencies_df = pd.DataFrame(columns=column_names), pd.DataFrame()
    for i in index_ids_streamer:
        info = roq.get_info(conn, i)
        if info != None:
            if info['index_id_streamer'] != None and info['index_id_player'] != None:
                # fetch rtp tables
                rtp_s = pd.read_sql(f"SELECT ts, seqnum_unwr FROM rtp WHERE index_id='{info['index_id_streamer']}' ORDER BY ts", 
                    conn, parse_dates=['ts'])
                rtp_p = pd.read_sql(f"SELECT ts, seqnum_unwr FROM rtp WHERE index_id='{info['index_id_player']}' ORDER BY ts", 
                    conn, parse_dates=['ts'])

                if not rtp_s.empty and not rtp_p.empty and info['air_ts'] != None:
                    air_time_s = (rtp_s["ts"] >= info['air_ts'][0]) & (rtp_s["ts"] <= info['air_ts'][1])
                    air_time_p = (rtp_p["ts"] >= info['air_ts'][0]) & (rtp_p["ts"] <= info['air_ts'][1])
                    rtp_merged = pd.merge(rtp_s[air_time_s], rtp_p[air_time_p], how='left', on='seqnum_unwr', suffixes=('_s', '_p')) # TODO why rtp_s[air_time_s] and rtp_p[air_time_p] have different column lengths? ask Hendrik. 
                    rtp_merged["ts_s"] = rtp_merged["ts_s"].apply(lambda t: t.replace(tzinfo=None))
                    rtp_merged["ts_p"] = rtp_merged["ts_p"].apply(lambda t: t.replace(tzinfo=None))
                    if not rtp_merged.empty:
                        network_latencies = roq.get_packet_latency(rtp_merged)
                        network_latencies = network_latencies.to_frame()
                        network_latencies = network_latencies.rename(columns= {0: 'latency'})
                        network_latencies.index.name = 'index'
                        network_latencies.index = rtp_merged["ts_s"]
                        network_latencies["index_id"] = len(network_latencies) * [i]
                        network_latencies_all = pd.concat([network_latencies_all, network_latencies])

    return network_latencies_all

def append_multi(list, data):
    for i in range(len(data)):
        list.append(data[i])
    
    return list

if __name__ == '__main__':
    main()
