#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <name> <port>"
    echo "Runs iperf3 on the server"
    exit 1
fi

NAME="$1"
PORT="$2"

LOG_DIR=/home/ubuntu/results/iperf_logs
mkdir -p "$LOG_DIR";

iperf3 --json -s -p ${PORT} --one-off --logfile "${LOG_DIR}/${NAME}.json"
