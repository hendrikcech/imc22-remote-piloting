#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <name> <interface> [<filter>]" >&2
    set +u
    echo "1:$1, 2:$2, 3:$3, 4:$4"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

NAME="$1"
IF="$2"
FILTER="${3:-}"
LIMIT_SIZE="${LIMIT_SIZE:-88}"

HOST="$(if [ -f '/boot/pi_id' ]; then echo 'pi'; else echo 'server'; fi)"

if [ "$HOST" = 'pi' ]; then
    PCAP_PATH='/home/pi/results/pcap'
    TCPDUMP_REPORT_PATH='/home/pi/results/tcpdump_report'
else
    PCAP_PATH='/home/ubuntu/results/pcap'
    TCPDUMP_REPORT_PATH='/home/ubuntu/results/tcpdump_report'
fi

echo "Start tcpdump on $HOST: $NAME" >&2

mkdir -p "$PCAP_PATH"
mkdir -p "$TCPDUMP_REPORT_PATH"

sudo tcpdump -i "$IF" -s "$LIMIT_SIZE" -B 16384 -w "$PCAP_PATH/${NAME}.pcap" -Z "$(whoami)" "$FILTER" \
    1>"${TCPDUMP_REPORT_PATH}/${NAME}.log" 2>&1 &
PID=$!

sleep 2

echo $PID
