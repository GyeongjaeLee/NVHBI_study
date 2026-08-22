#!/bin/bash
# run_exp1_bisection.sh -- EXPERIMENT 1
#
# NV-HBI (die-to-die) write bisection bandwidth vs the number of writing SMs, on
# one B200. Produces plot 1a: x = num_active_sm, y = slope_GBps, one curve per
# own_die value and per block_size.
#
# Measured with two-point slope timing plus an in-kernel clock readout, not with
# ncu. ncu needs GPU performance-counter permission (ERR_NVGPUCTRPERM), which
# containers do not have, and the slope method removes the fixed cost that ncu
# was wanted for in the first place. See nvhbi_exp1_bisection.cu for the details.
#
# Result on B200 (74 SMs, cross-die): 4753 GB/s +/- 4% across three sessions.
# The own_die curve is what makes that a fabric number: cross-die and own-die
# agree to 4 significant figures up to 32 SMs, then only cross-die flattens
# (own/cross = 1.18 at 64 SMs, 1.33 at 77).
#
# Usage:
#   ./run_exp1_bisection.sh                       # both partitions, cross + own
#   ./run_exp1_bisection.sh 1                     # partition 1 only
#   NVHBI_SMS=74 NVHBI_BLOCK_SIZES=64 NVHBI_OWN_DIE=0 \
#     ./run_exp1_bisection.sh 1                   # single point, for repeats
#
# Env: GPU_ID CSV_OUT + every NVHBI_* knob the binary reads.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

GPU_ID="${GPU_ID:-0}"
PROG="nvhbi_exp1_bisection"
CSV_OUT="${CSV_OUT:-exp1_bisection.csv}"
PARTITION="${1:-2}"

CFG_HEADER="writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,footprint_MB,sectors_lo,sectors_hi,ms_lo,ms_hi,naive_GBps,slope_GBps,counted_GBps,count_ratio,overhead_ms,eff_GHz,span_ratio,sampled_GBps,sample_ms"

nvhbi_detect_gpu "$GPU_ID"
nvhbi_build "$PROG"

nvhbi_clock_snapshot "before" "$GPU_ID"
{
  echo "$CFG_HEADER"
  CUDA_VISIBLE_DEVICES="$GPU_ID" ./"$PROG" "$PARTITION" \
    | awk '/^CFG,/ { sub(/^CFG,/, ""); print }'
} > "$CSV_OUT"
nvhbi_clock_snapshot "after " "$GPU_ID"

rows=$(( $(wc -l < "$CSV_OUT" | tr -d ' ') - 1 ))
echo "Done. Wrote $CSV_OUT ($rows rows)"

cat <<'EOF'

How to read it
--------------
slope_GBps  the measurement: d(sectors x 32B)/d(time) across the two iteration
            counts, so launch, ramp-up, first-touch and tail all cancel.
naive_GBps  bytes/time of the long run, i.e. what a single-shot measurement
            would have said. Should now sit within a few percent of slope_GBps.
overhead_ms fixed cost the two points imply. Should be ~0; if it grows past ~10%
            of ms_hi, raise NVHBI_ITER_HI.
eff_GHz     SM clock actually achieved. Only compare bandwidth between rows at
            the same eff_GHz -- identical work once measured 1.56x apart across
            sessions before this was tracked. It is read from one designated
            thread, so it also reflects load imbalance: compare within an SM
            count, not across SM counts.

Plot 1a: x = num_active_sm (log), y = slope_GBps, series = own_die x block_size.
The cross-die and own-die curves overlapping and then separating IS the argument.
EOF
