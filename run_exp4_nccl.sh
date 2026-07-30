#!/bin/bash
# run_exp4_nccl.sh -- EXPERIMENT 4
#
# NCCL all-to-all across 2 GPUs, swept over message size and background load,
# run twice: with a background that crosses the dies and with one that does not.
#
#   NVHBI_BG_LOCAL=0  background loads the bisection
#   NVHBI_BG_LOCAL=1  same SMs, same store rate, nothing crosses -> CONTROL
#
# The difference between the two is the bisection effect. Without the control,
# "NCCL got slower" is indistinguishable from NCCL simply losing SMs, because
# NCCL's own kernels run on GPU0 too.
#
# Two limits to state when reporting:
#   * NCCL buffers are ordinary cudaMalloc and die ownership follows the physical
#     address, so the collective's traffic lands ~50/50 on both dies. This is a
#     contention study, not a die-targeted one (exp 2/3 are die-targeted).
#   * 2 ranks makes all-to-all a pairwise exchange. Use 4 or 8 GPUs for a real one.
#
# Usage:
#   ./run_exp4_nccl.sh
#   NVHBI_NCCL_BYTES_LIST=1048576,8388608,67108864,268435456 ./run_exp4_nccl.sh
#   NVHBI_BG_SMS_LIST=0,8,16,32,64,74 ./run_exp4_nccl.sh
#
# Env: CSV_OUT NCCL_HOME + every NVHBI_* knob in nvhbi_exp4_nccl.cu

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

PROG="nvhbi_exp4_nccl"
CSV_OUT="${CSV_OUT:-exp4_nccl.csv}"
LOG_OUT="${LOG_OUT:-exp4_nccl.log}"
CFG_FIELDS=9
CFG_HEADER="exp,bg_local,bg_sms,msg_bytes,rep,nccl_ms,algbw_GBps,busbw_GBps,bg_GBps"

ndev=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$ndev" -lt 2 ]]; then
  echo "ERROR: this experiment needs 2 GPUs, found $ndev" >&2
  exit 1
fi

nvhbi_detect_gpu 0
if ! nvhbi_build "$PROG"; then
  echo >&2
  echo "Build failed. If NCCL headers/libs were not found, point NCCL_HOME at them:" >&2
  echo "  NCCL_HOME=/path/to/nccl ./run_exp4_nccl.sh" >&2
  exit 1
fi

: > "$LOG_OUT"
for local in 0 1; do
  echo "=== bg_local=$local ($([[ $local == 0 ]] && echo 'cross-die: loads the bisection' || echo 'own-die: CONTROL')) ==="
  NVHBI_BG_LOCAL="$local" ./"$PROG" 2>&1 | tee -a "$LOG_OUT"
  echo
done

{
  echo "$CFG_HEADER"
  nvhbi_cfg_rows "$LOG_OUT" "$CFG_FIELDS"
} > "$CSV_OUT"

rows=$(( $(wc -l < "$CSV_OUT" | tr -d ' ') - 1 ))
echo "Done. Wrote $CSV_OUT ($rows rows), full output in $LOG_OUT"
cat <<'EOF'

How to read it
--------------
Main plot     x = msg_bytes (log)   y = busbw_GBps
              series: bg_sms = 0 / 16 / 32 / 64, solid for bg_local=0
              and dashed for bg_local=1. The gap between solid and dashed at
              the same bg_sms is the bisection effect; the drop from bg_sms=0
              that both share is SM contention.

Trade plot    x = bg_sms            y = busbw_GBps (left) and bg_GBps (right)
              at the largest message size. Shows what each side gives up.
EOF
