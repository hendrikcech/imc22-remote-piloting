#!/usr/bin/env bash
set -eu

usage() {
    echo "Usage: $(basename "$0") <interface|provider|rt_nr_from_provider> <value>" 1>&2
    echo "Usage: $(basename "$0") <pi_id|check>" 1>&2
    exit 1
}
if [ $# -eq 1 ]; then
    case "$1" in
        pi_id)
            cat /boot/pi_id
            ;;
        summary)
            RETURN=0
            for PROVIDER in P1 P2; do
                echo "$PROVIDER = $($0 interface $PROVIDER) = RT #$($0 rt_nr_from_provider $PROVIDER)"
                # Makes no sense right now with only one interface per Pi
                # if [ "$PROVIDER" != "$(info.sh provider $(info.sh interface $PROVIDER))" ]; then
                #     echo "Mapping for $PROVIDER inconsistent!"
                #     RETURN=1
                # fi
            done
            exit $RETURN
            ;;
        aws_ip)
            sudo grep Endpoint /etc/wireguard/wg_pi.conf | \
                cut -d '=' -f 2 | cut -d ':' -f 1 | xargs
            ;;
        *)
            usage
        esac
    exit 0
fi

if [ $# -ne 2 ]; then
    usage
fi
QUERY="$1"
VALUE="$2"

MODEM_LOG_DIR='/home/pi/results/ppp'
LINE="$(grep --no-filename --text "$VALUE" $MODEM_LOG_DIR/* | sort | tail -n 1)"

return_fn() {
    if [ -z "${1:-}" ]; then
        exit 1
    else
        echo "$1"
        exit 0
    fi
}

case "$QUERY" in
    interface) # -> ppp1
        return_fn "$(echo "$LINE" | cut -d ';' -f 4)"
        ;;
    provider) # -> P2
        return_fn "$(echo "$LINE" | cut -d ';' -f 3)"
        ;;
    rt_nr_from_provider) # -> 1 or 2
        return_fn "$(grep "$VALUE" /etc/iproute2/rt_tables | cut -d ' ' -f 1)"
        ;;
    *)
        usage
    esac
