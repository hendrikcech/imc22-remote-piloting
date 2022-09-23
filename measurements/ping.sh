#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "$0 <icmp|tcp>"
    echo "Runs a ping measurement and stores the results"
    exit 1
fi
METHOD="$1"

LOG_DIR='/home/pi/results/ping'
TS="$(date -u '+%Y%m%dT%H%M%S')"
PROVIDER="$(info.sh provider ppp0)"
NAME="${TS}_ping_${METHOD}_$(cat /boot/pi_id)_${PROVIDER}"

mkdir -p "$LOG_DIR"

echo "tcpdump on server running?"
echo "sudo tcpdump -i eth0 icmp or tcp port 80 -w /home/ubuntu/results/${TS}_pings.pcap"

trap_fn() {
    sudo pkill -ef "tcpdump.*${NAME}"
}
trap trap_fn EXIT TERM INT

if [ "$METHOD" = 'icmp' ]; then
    TCPDUMP_FILTER='icmp'
elif [ "$METHOD" = 'tcp' ]; then
    TCPDUMP_FILTER='tcp port 80'
fi
tcpdump.sh "$NAME" ppp0 "$TCPDUMP_FILTER"

if [ "$METHOD" = 'icmp' ]; then
    sudo stdbuf -oL ping -i .2 pi-server.cech.io | \
        while read -r LINE; do
            echo "$(date -u '+%Y-%m-%dT%H:%M:%S.%NZ');${LINE}"
        done > "${LOG_DIR}/${NAME}.txt"
elif [ "$METHOD" = 'tcp' ]; then
    sudo stdbuf -oL hping3 -S -i u500000 -p 80 pi-server.cech.io | \
        while read -r LINE; do
            echo "$(date -u '+%Y-%m-%dT%H:%M:%S.%NZ');${LINE}"
        done > "${LOG_DIR}/${TS}_$(cat /boot/pi_id)_ping_tcp.txt"
fi
