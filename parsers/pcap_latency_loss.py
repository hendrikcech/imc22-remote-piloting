#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
import os
from scapy.all import Raw, bind_layers
from scapy.utils import PcapReader, RawPcapReader
from scapy.layers.inet import UDP, ICMP
from scapy.layers.rtp import RTP

def parse_iperf_udp_seq(packet):
    payload = packet.payload.load
    if len(payload) < 12:
        return None
    # iperf_sec = int.from_bytes(payload[0:4], byteorder='big')
    # iperf_usec = int.from_bytes(payload[4:8], byteorder='big')
    iperf_seq = int.from_bytes(payload[8:12], byteorder='big')
    return iperf_seq

def get_rtp_parser():
    rtp_generation = {}
    rtp_last_seq = {}
    def parse_rtp_seq(port, packet):
        packet_rtp = RTP(packet["Raw"].load[1:])
        if packet_rtp.sequence == 0 and packet_rtp.version == 0:
            return None
        if port not in rtp_generation:
            rtp_generation[port] = 0
        if port not in rtp_last_seq:
            rtp_last_seq[port] = packet_rtp.sequence
        if rtp_last_seq[port] - packet_rtp.sequence > 1000:
            # Wrap around detected
            rtp_generation[port] += 1
        if packet_rtp.sequence > 30000 and rtp_last_seq[port] < 5000: # out-of-order packets; correct wrap-around
            rtp_generation[port] -= 1
        rtp_last_seq[port] = packet_rtp.sequence
        # if packet_rtp.sequence == 0:
        #     breakpoint()
        return packet_rtp.sequence + rtp_generation[port] * 2**16
    return parse_rtp_seq

def format_unix_ts(ts):
    return pd.to_datetime(float(ts), unit='s', origin='unix', utc=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse pcaps and calculate the packet latencies.')
    parser.add_argument('sndr_pcap',  help='Sender pcap')
    parser.add_argument('rcvr_pcap',  help='Receiver pcap')
    # parser.add_argument('path_prefix',  help='Path prefix to pcaps. Will be extended to PATH_pi.pcap and PATH_server.pcap')
    # parser.add_argument('out',  help='Writes the result to this csv file')
    parser.add_argument('--break-after', help='Only process N packets of each pcap', default=0, type=int)
    args = parser.parse_args()

    # sndr_pcap = args.path_prefix + '_pi.pcap'
    # rcvr_pcap = args.path_prefix + '_server.pcap'
    # out_latency = args.path_prefix + '_pi.latency.csv'
    # out_losses = args.path_prefix + '_pi.loss.csv'

    sndr_pcap = args.sndr_pcap
    rcvr_pcap = args.rcvr_pcap
    folder = os.path.dirname(sndr_pcap)
    name = os.path.splitext(os.path.basename(sndr_pcap))[0]
    out_latency = os.path.join(folder, name + ".latency.csv")
    out_losses = os.path.join(folder, name + ".loss.csv")

    print(f"Parsing {sndr_pcap=} and {rcvr_pcap=} to {out_latency=} and {out_losses=}")

    iperf_ports = [7011, 7022]
    rtp_ports = [6001, 6002, 6003, 6004]
    ports = iperf_ports + rtp_ports + [1]

    # for port in rtp_ports:
    #     bind_layers(UDP, RTP, dport=port)

    sent = { port: {} for port in ports } # {port: {seq: ts}}
    received = { port: set() for port in ports } # {port: {seq set}}
    # latency = { port: {} for port in ports } # {port: {seq: ts}}

    parse_sndr_rtp_seq = get_rtp_parser()
    parse_rcvr_rtp_seq = get_rtp_parser()
    for packet_nr, packet in enumerate(PcapReader(sndr_pcap)):
        sys.stdout.write(f"\rReading sender packet #{packet_nr+1}")

        seq = None
        port = None
        if UDP in packet:
            port = packet[UDP].dport
            if port in iperf_ports:
                seq = parse_iperf_udp_seq(packet)
            elif port in rtp_ports:
                seq = parse_sndr_rtp_seq(port, packet)
        elif ICMP in packet and packet[ICMP].type == 8: # type==echo-request
            seq = packet[ICMP].seq
            port = 1 # ICMP ping computation only works with one ICMP source in server pcap
        else:
            continue

        if seq is None:
            continue

        if seq in sent[port]:
            print(f"Packet with {seq=} sent twice")
            breakpoint()

        sent[port][seq] = packet.time # unix time

        if args.break_after > 0 and packet_nr >= args.break_after:
            breakpoint()
            break

    print() # newline

    latency = []
    last_packet_received_ts = None
    for packet_nr, packet in enumerate(PcapReader(rcvr_pcap)):
        sys.stdout.write(f"\rReading receiver packet #{packet_nr+1}")

        seq = None
        port = None
        if UDP in packet:
            port = packet[UDP].dport
            if port in iperf_ports:
                seq = parse_iperf_udp_seq(packet)
            elif port in rtp_ports:
                seq = parse_rcvr_rtp_seq(port, packet)
        elif ICMP in packet and packet[ICMP].type == 8: # type==echo-request
            seq = packet[ICMP].seq
            port = 1
        else:
            continue

        last_packet_received_ts = packet.time

        if seq is None:
            continue

        if seq not in sent[port]:
            # print(f"Packet {iperf_seq=} received that was not sent")
            pass
        else:
            if seq in received[port]:
                print(f"Packet {seq=} received twice")
            received[port].add(seq)
            sent_ts_unix = sent[port][seq]
            ts_sent = format_unix_ts(sent_ts_unix)
            ts_rcvd = format_unix_ts(packet.time)
            packet_latency_ms = (packet.time - sent[port][seq]) * 1000 # ms
            if packet_latency_ms < 0:
                print(f"Calculated latency is negative")
                breakpoint()
            latency.append((ts_sent, ts_rcvd, port, seq, float(packet_latency_ms)))

        if args.break_after > 0 and packet_nr >= args.break_after:
            breakpoint()
            break

    print() # newline

    df_latency = pd.DataFrame(latency, columns=['ts_sent', 'ts_rcvd', 'dst_port', 'seq_unwr', 'latency_ms'])
    df_latency.to_csv(out_latency, sep='\t', index=False, float_format='%.3f')

    losses = []
    for port, seqs_sent in sent.items():
        for seq, ts_sent in seqs_sent.items():
            # if ts_sent - last_packet_received_ts < 5:
            #     # Don't compute packet losses that happend in the last seconds of the tests. Some packets may still have been-inflight
            #     continue
            if not seq in received[port]:
                losses.append((format_unix_ts(ts_sent), port, seq))

    df_loss = pd.DataFrame(losses, columns=['ts_sent', 'dst_port', 'seq_unwr'])
    df_loss.to_csv(out_losses, sep='\t', index=False, float_format='%.3f')
