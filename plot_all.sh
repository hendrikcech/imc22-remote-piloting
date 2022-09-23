#!/usr/bin/env sh

set -eu

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <path to sqlite database>"
    echo "Fetch database with measurement data from TODO"
    exit 1
fi
DB="$1"

SCRIPT_DIR="$(dirname "$0")/figures"
OUTPUT_DIR="$(dirname "$0")/figures_pdf"
mkdir -p "$OUTPUT_DIR"

set -x

python3 "$SCRIPT_DIR/04a-handover_frequency.py" --save "$OUTPUT_DIR/04a-handover_frequency.pdf"
python3 "$SCRIPT_DIR/04b-het_duration.py" "$DB" --save "$OUTPUT_DIR/04b-het_duration.pdf"
python3 "$SCRIPT_DIR/05-latency.py" "$DB" --save "$OUTPUT_DIR/05-latency.pdf"
python3 "$SCRIPT_DIR/06-07-12-video_metrics.py" "$DB" 'urban' --save "$OUTPUT_DIR/06-07-video_metrics.pdf"
python3 "$SCRIPT_DIR/06-07-12-video_metrics.py" "$DB" 'rural' --save "$OUTPUT_DIR/12-video_metrics.pdf"
python3 "$SCRIPT_DIR/08a-handover_latency.py" "$DB" --save "$OUTPUT_DIR/08a-handover_latency.pdf"
python3 "$SCRIPT_DIR/08b-handover_latency.py" "$DB" --save "$OUTPUT_DIR/08b-handover_latency.pdf"
python3 "$SCRIPT_DIR/09-latency_handover_analysis.py" "$DB" --save "$OUTPUT_DIR/09-latency_handover_analysis.pdf"
python3 "$SCRIPT_DIR/10a-throughput.py" --save "$OUTPUT_DIR/10a-throughput.pdf"
python3 "$SCRIPT_DIR/10b-handover_frequency.py" --save "$OUTPUT_DIR/10b-handover_frequency.pdf"
python3 "$SCRIPT_DIR/13-latency_altitude.py" "$DB" --save "$OUTPUT_DIR/13-latency_altitude.pdf"

set +x
