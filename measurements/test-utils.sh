#!/usr/bin/env bash
set -u

start_tcpdump() {
    NAME="$1"
    IF="$2"
    ssh aws tcpdump.sh "${NAME}_server" any 2>&1 | grep -v '^Load key.*Permission denied$'
    tcpdump.sh "${NAME}_pi" "$IF"
}

stop_tcpdump() {
    NAME="${1:-"$NAME"}"
    sudo pkill -ef "tcpdump.*${NAME}"
    ssh aws "sudo pkill -ef 'tcpdump.*${NAME}'"
}

stop_tcpdump_pkill() {
    NAME="$1"
    sudo pkill -ef "tcpdump.*${NAME}"
    ssh aws "sudo pkill -ef 'tcpdump.*${NAME}'"
}

start_tcpinfo() {
    NAME="$1"
    SELECTOR="$2"
    tcp_info.sh "${NAME}_pi" "${SELECTOR}" &
    ssh aws "tcp_info.sh '${NAME}_server'" &
}

stop_tcpinfo() {
    NAME="${1:-"$NAME"}"
    sudo pkill -ef "tcp_info.*${NAME}"
    ssh aws "sudo pkill -ef 'tcp_info.*${NAME}'"
}

get_name() {
    NAME="$(date +%Y%m%dT%H%M%S)_${LABEL}_${TYPE}_${PROTO}_${DIRECTION}_$(info.sh pi_id)_$(info.sh provider "$IF")"
}

block_until_ctrlc() {
    while true; do
        if ! read -p "Use Ctrl-C to stop the test" INPUT; then
            while true; do sleep 1; done
        fi
        break
    done;
}

wait_for_stop() {
    while true; do
        read -p "Enter 'stop' to stop the test" INPUT
        if [ "$INPUT" == 'stop' ]; then
            echo "Stopping the test"
            break;
        fi
    done
}
