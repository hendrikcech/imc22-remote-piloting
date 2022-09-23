#!/usr/bin/env bash

set -eu

# Start tcpdump and ROQ with the chosen CC (gcc, scream, or static)

source test-utils.sh

ROQ_CC="$ROQ_CC"
ROQ_DL="${ROQ_DL:-0}"
if [ "$ROQ_CC" = 'static' ]; then
    ROQ_STATIC_RATE="$ROQ_STATIC_RATE"
else
    ROQ_STATIC_RATE=''
fi
LABEL="${LABEL:-"220313T"}"

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
NAME_TIMESTAMP="${TIMESTAMP}_${LABEL}"

trap_fn() {
    set +e +u
    STATUS=$?
    pkill -ef "roq_serve.*${ROQ_CC}" || true
    if [ "$ROQ_DL" -gt  0 ]; then
        pkill -ef "iperf_sp.*" || true
    fi
    stop_tcpdump "$PCAP_NAME"
    return $STATUS
}
trap trap_fn EXIT TERM INT

PORT="600$(info.sh pi_id)"
PCAP_NAME="${TIMESTAMP}_${LABEL}_$(info.sh pi_id)"
# start_tcpdump ".rtp" udp dst port $PORT
ssh aws "tcpdump.sh \"${PCAP_NAME}_server.rtp\" eth0 \"udp dst port $PORT\"" 2>&1 | grep -v '^Load key.*Permission denied$'
         tcpdump.sh "${PCAP_NAME}_pi.rtp"     ppp0 "udp dst port $PORT"
ssh aws "LIMIT_SIZE=0 tcpdump.sh \"${PCAP_NAME}_server.rtcp\" eth0 \"udp src port $PORT\"" 2>&1 | grep -v '^Load key.*Permission denied$'
         LIMIT_SIZE=0 tcpdump.sh "${PCAP_NAME}_pi.rtcp"     ppp0 "udp src port $PORT"

(sleep 2; NAME_TIMESTAMP="$NAME_TIMESTAMP" PORT="$PORT" roq_serve.sh udp "$ROQ_CC" "$ROQ_STATIC_RATE") &

if [ "$ROQ_DL" -gt  0 ]; then
    PI_ID="$(cat /boot/pi_id)"
    PROVIDER="$(info.sh provider "ppp0")"
    DIRECTION_DIGIT=0
    RT_NR="$(info.sh rt_nr_from_provider "$PROVIDER")"
    IPERF_PORT="7${DIRECTION_DIGIT}${PI_ID}${RT_NR}"
    echo "IPERF_PORT: $IPERF_PORT"
    ssh aws "tcpdump.sh \"${PCAP_NAME}_server.iperf\" eth0 \"udp dst port $IPERF_PORT\"" 2>&1 | grep -v '^Load key.*Permission denied$'
            tcpdump.sh "${PCAP_NAME}_pi.iperf"     ppp0 "udp dst port $IPERF_PORT"
    (sleep 2; NAME_TIMESTAMP="$NAME_TIMESTAMP" iperf_sp.sh ppp0 udp down "${ROQ_DL}M") &
fi

wait
