#!/usr/bin/env bash
# Builds all data for the shared potholes + signs detector.
# Usage: bash detector/prepare_shared.sh RAW_RDD SIGN_SRC RDD_OUT SIGN_OUT
set -euo pipefail
RAW_RDD="$1"; SIGN_SRC="$2"; RDD_OUT="$3"; SIGN_OUT="$4"
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "== 1/3 RDD2022 -> YOLO"
python "$HERE/rdd_prepare.py" --src "$RAW_RDD" --out "$RDD_OUT"
echo "== 2/3 Signs -> our classes, close-up occluded sets"
python "$HERE/signs_prepare.py" --src "$SIGN_SRC" --out "$SIGN_OUT" --rdd "$RDD_OUT"
echo "== 3/3 Signs pasted into road scenes at realistic sizes"
python "$HERE/signs_paste.py" --signs "$SIGN_OUT" --rdd "$RDD_OUT"
test -f "$SIGN_OUT/data_combined.yaml"
echo "== Done: $SIGN_OUT/data_combined.yaml"
