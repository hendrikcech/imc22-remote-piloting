#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <name> <port> <udp|tcp|quic> <gcc|scream>"
    echo "Runs on the server and plays a video stream"
    exit 1
}

if [ -z "${1-}" ]; then
    usage
fi
NAME="$1"
PORT="$2"
PROTOCOL="$3"
RTP_CC="$4"
if [ "$(grep -E -c '^(tcp|udp)$' <<< "$PROTOCOL")" -ne 1 ]; then
    echo "Invalid protocol '$PROTOCOL'"
    usage
fi
if [ "$(grep -E -c '^(gcc|scream|static)$' <<< "$RTP_CC")" -ne 1 ]; then
    echo "Invalid RTP CC '$RTP_CC'"
    usage
fi

LOG_DIR="$HOME/results/roq"
mkdir -p "$LOG_DIR"

echo "Listen on 0.0.0.0:$PORT for $NAME"

if [ "$RTP_CC" == 'gcc' ]; then
    RTCP_FORMAT='--twcc'
elif [ "$RTP_CC" == 'scream' ]; then
    RTCP_FORMAT='--rfc8888'
elif [ "$RTP_CC" == 'static' ]; then
    RTCP_FORMAT=''
fi

set -x

GST_DEBUG=*:3,rtpjitterbuffer:4,timecodeparse:4 \
stdbuf -oL -eL \
roq receive -a ":${PORT}" --sink fpsdisplaysink \
    --rtp-dump "${LOG_DIR}/${NAME}.rtp.csv" --fps-dump "${LOG_DIR}/${NAME}.fps.csv" \
    --rtcp-dump "${LOG_DIR}/${NAME}.rtcp.csv" \
    --save "${LOG_DIR}/${NAME}.avi" \
    --transport "${PROTOCOL}" \
    ${RTCP_FORMAT} \
    2>&1 | tee "${LOG_DIR}/${NAME}.log"

set +x
