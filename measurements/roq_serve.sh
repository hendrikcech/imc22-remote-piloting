#!/usr/bin/env bash

set -eu

usage() {
    echo "Usage: $0 <protocol: udp|tcp|quic> <CC: gcc or scream>"
    echo "Starts a rtp-over-quic video delivery. SSHs to the server and starts roq_play.sh there"
    exit 1
}

if [ -z "${1-}" ]; then
    usage
fi
PROTOCOL="$1"
RTP_CC="$2"
if [ "$(grep -E -c '^(tcp|udp)$' <<< "$PROTOCOL")" -ne 1 ]; then
    echo "Invalid protocol '$PROTOCOL'"
    usage
fi
if [ "$(grep -E -c '^(gcc|scream|static)$' <<< "$RTP_CC")" -ne 1 ]; then
    echo "Invalid RTP CC '$RTP_CC'"
    usage
fi

if [ "$RTP_CC" = 'static' ]; then
    RTP_STATIC_RATE="$3"
fi

PI_ID="$(info.sh pi_id)"
PORT="${PORT:-600${PI_ID}}"
PROVIDER="$(info.sh provider ppp0)"
if [ -z "${NAME:-}" ]; then
    TIMESTAMP="${NAME_TIMESTAMP:-$(date +%Y%m%dT%H%M%S)}"
    NAME="${TIMESTAMP}_roq_${RTP_CC}_${PI_ID}_${PROVIDER}"
fi
NAME_PI="${NAME}_pi"
NAME_SERVER="${NAME}_server"

HOST="$(grep -A 1 'Host aws' /etc/ssh/ssh_config | grep Hostname | xargs | cut -d ' ' -f 2)"
echo "${PROVIDER}, pi #${PI_ID}: RTP over ${PROTOCOL} with ${RTP_CC} to $HOST:$PORT"

LOG_DIR="${LOG_DIR:-$HOME/results/roq}"
mkdir -p "$LOG_DIR"

export LIBVA_DRIVER_NAME=i965
sudo ip link set ppp0 mtu 1500

cleanup() {
    pkill -f "roq.*$PORT" || true
    sleep 3
    ssh aws "pkill -f 'roq.*${PORT}'" || true
    sleep 3
    pkill -9 -f "roq.*$PORT" || true
    ssh aws "pkill -9 -f 'roq.*${PORT}'" || true
}
cleanup
trap cleanup EXIT

ssh aws "roq_play.sh $NAME_SERVER $PORT $PROTOCOL $RTP_CC" >/dev/null 2>&1 &
# Player needs to be up before streamer
sleep 13


RTP_CC_ARG="--$RTP_CC"
if [ "$RTP_CC" = 'static' ]; then
    RTP_CC_ARG="--init-rate ${RTP_STATIC_RATE}"
fi

SOURCE=${ROQ_SOURCE:-"$HOME/train_30.mp4"}
echo "Serving $SOURCE"

GST_DEBUG=*:3,rtpjitterbuffer:4,timecodeoverlay:4 \
stdbuf -oL -eL \
roq send -a "${HOST}:${PORT}" --source "$SOURCE" --codec h264 \
    --rtp-dump "${LOG_DIR}/${NAME_PI}.rtp.csv" --cc-dump "${LOG_DIR}/${NAME_PI}.cc.csv" \
    --rtcp-dump "${LOG_DIR}/${NAME_PI}.rtcp.csv" \
    --save "${LOG_DIR}/${NAME_PI}.avi" \
    --transport "${PROTOCOL}" \
    ${RTP_CC_ARG} \
    2>&1 | tee "${LOG_DIR}/${NAME_PI}.log"
