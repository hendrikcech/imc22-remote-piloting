#!/usr/bin/env bash
set -uo pipefail

usage() {
    echo "Usage: $0 <interface> <tcp|udp> <up|down> [target bitrate]"
    echo "Start an iperf3 transfer from the client"
    exit 1
}

if [ "$#" -lt 3 ]; then
    usage
fi
IF="$1"
PROTOCOL="$2"
if [ "$(grep -E -c '^(tcp|udp)$' <<< "$PROTOCOL")" -ne 1 ]; then
    echo "Invalid protocol '$PROTOCOL'"
    usage
fi
DIRECTION="$3"
if [ "$(grep -E -c '^(up|down)$' <<< "$DIRECTION")" -ne 1 ]; then
    echo "Invalid direction '$DIRECTION'"
    usage
fi
BITRATE="${4-0}"

PI_ID="$(cat /boot/pi_id)"
if [ -z "$PI_ID" ]; then
    echo "Can't find Pi ID in /boot/pi_id!"
    exit 1
fi
IF_IP="$(ip addr show "$IF" | grep -Po 'inet \K[\d.]+')"
if [ -z "$IF_IP" ]; then
    echo "Can't find IP of $IF"
    exit 1
fi
HOST="$(grep -A 1 'Host aws' /etc/ssh/ssh_config | grep Hostname | xargs | cut -d ' ' -f 2)"
LOG_PATH='/home/pi/results/iperf_logs'
PROBE_PATH='/home/pi/results/probe'
mkdir -p "$LOG_PATH"
mkdir -p "$PROBE_PATH"
PROVIDER="$(info.sh provider "$IF")"
if [ -z "${NAME:-}" ]; then
    TIMESTAMP="${NAME_TIMESTAMP:-$(date +%Y%m%dT%H%M%S)}"
    NAME="${TIMESTAMP}_iperf_${PROTOCOL}_${DIRECTION}_${PI_ID}_${PROVIDER}"
fi
NAME_PI="${NAME}_pi"
NAME_SERVER="${NAME}_server"

if [ "$DIRECTION" = 'up' ]; then
    DIRECTION_DIGIT=0
elif [ "$DIRECTION" = 'down' ]; then
    DIRECTION_DIGIT=1
fi
RT_NR="$(info.sh rt_nr_from_provider "$PROVIDER")"
if [ -z "${RT_NR:-}" ]; then
    RT_NR=0
fi
IPERF_PORT="7${DIRECTION_DIGIT}${PI_ID}${RT_NR}"

sudo sysctl -w net.mptcp.mptcp_enabled=0

cleanup() {
    STATUS=$?
    ssh aws "pkill -f 'iperf_server.sh.*${NAME_SERVER}'" || true
    return $STATUS
}

ssh aws "iperf_server.sh '$NAME_SERVER' '$IPERF_PORT'" &
trap cleanup EXIT

echo "Start iperf to port ${IPERF_PORT}: $NAME"
sleep 2

if [ "$PROTOCOL" = 'tcp' ]; then
    echo 'Reminder: tcp_probe.sh running?'
fi

OPTIONS=''
if [ "$PROTOCOL" == 'udp' ]; then
    OPTIONS="${OPTIONS} -u"
fi
if [ "$DIRECTION" == 'down' ]; then
    OPTIONS="${OPTIONS} -R"
fi

iperf3 --json -c "$HOST" -p "$IPERF_PORT" -t 36000 --bind "$IF_IP" -b "$BITRATE" $OPTIONS --logfile "$LOG_PATH/${NAME_PI}.json"

sleep 2
