#!/bin/bash
# A drop-in "python" for sec's start_kev.sh (KEV_PYTHON=this file): runs `kev_patch.py SCRIPT ARGS...` with the real
# interpreter KEV_REAL_PYTHON. With KEV_FP8_TUNED=1 the served model's FP8Linear GEMMs use the confirmed bitwise table
# (KEV_FP8_TUNED_TABLE, default tables/ada_rtx6000_bitwise_r1.json); otherwise the launcher changes nothing.
set -eu
: "${KEV_REAL_PYTHON:?set KEV_REAL_PYTHON to the production venv python}"
exec "$KEV_REAL_PYTHON" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kev_patch.py" "$@"
