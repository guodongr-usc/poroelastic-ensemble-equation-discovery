#!/bin/bash
# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

set -euo pipefail
[[ $# == 4 ]] || { echo 'Expected CONFIG PYTHON ROOT MODE' >&2; exit 2; }
CONFIG="$1"; PY="$2"; ROOT="$3"; MODE="$4"
cd -- "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset PYTHONPATH PYTHONHOME JAX_PLATFORMS JAX_PLATFORM_NAME XLA_FLAGS
printf 'MODE=%s JOB=%s TASK=%s HOST=%s ALLOC_CPUS=%s\n' "$MODE" "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "${SLURM_CPUS_PER_TASK:-none}"
args=(--config "$CONFIG" --mode "$MODE")
if [[ "$MODE" == coarse || "$MODE" == fine ]]; then args+=(--trial "${SLURM_ARRAY_TASK_ID:?array index missing}"); fi
set +e
"$PY" -s -u "$ROOT/workflow.py" "${args[@]}"
rc=$?
set -e
if [[ "$rc" == 75 ]]; then
 echo 'Deadline reached; checkpoint retained. Use submit.py --resume after jobs stop.'
fi
exit "$rc"
