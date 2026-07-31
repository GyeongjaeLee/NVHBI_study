#!/bin/bash
# run_exp23_peer.sh -- EXPERIMENTS 2 and 3
#
# GPU1 pushes NVLink traffic into GPU0 while GPU0 runs the experiment-1
# remote-L2 write load. NVLink is assumed to land on GPU0's die 0, so:
#
#   exp2 (target die 1) -- peer traffic also crosses NV-HBI
#   exp3 (target die 0) -- same NVLink load, no crossing: the CONTROL
#
# The binary prints a PRE-CHECK of die-0 vs die-1 peer read latency. If those
# are equal, NVLink is not die-0-attached and the exp2/exp3 contrast is void.
# Read that before anything else.
#
# The measured quantity is the BACKGROUND bandwidth (bg_GBps): it is 100%
# bisection-crossing and is the same traffic exp1 calibrated with ncu, so it
# acts as a probe of how much bisection is still available. The foreground is
# the independent variable, swept over grid size.
#
# Four runs by default: {exp2, exp3} x {cross-die background, own-die background}.
# The own-die background is the control that separates fabric contention from
# plain SM/L2 contention.
#
# Usage:
#   ./run_exp23_peer.sh                     # all four runs
#   ./run_exp23_peer.sh 2                   # exp2 only (both background modes)
#   NVHBI_BG_SMS_LIST=0,8,16,32,64,74 \
#   NVHBI_PEER_BLOCKS_LIST=0,64,256,1024,4096 ./run_exp23_peer.sh
#   NVHBI_MEMCPY=1 ./run_exp23_peer.sh      # also time cudaMemcpyPeer
#
# Two knobs worth running before believing a flat result:
#   NVHBI_PEER_IDLE=<cycles>   Throttles injection continuously. The default
#                              sweep saturates NVLink at its smallest grid, so
#                              without this there are no sub-saturation points.
#   NVHBI_PEER_OVERLAP=1       Peer and background write the SAME chunks instead
#                              of disjoint regions.
#
# Env: CSV_OUT + every NVHBI_* knob in nvhbi_exp23_peer.cu

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

PROG="nvhbi_exp23_peer"
CSV_OUT="${CSV_OUT:-exp23_peer.csv}"
LOG_OUT="${LOG_OUT:-exp23_peer.log}"
WHICH="${1:-both}"
CFG_FIELDS=13
CFG_HEADER="exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl"

ndev=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$ndev" -lt 2 ]]; then
  echo "ERROR: this experiment needs 2 GPUs, found $ndev" >&2
  exit 1
fi

nvhbi_detect_gpu 0
nvhbi_build "$PROG"

echo "P2P topology:"
nvidia-smi topo -m 2>/dev/null | head -20 || true
echo

case "$WHICH" in
  2)    EXPS=(2) ;;
  3)    EXPS=(3) ;;
  both) EXPS=(2 3) ;;
  *)    echo "usage: $0 [2|3|both]" >&2; exit 1 ;;
esac

: > "$LOG_OUT"
for e in "${EXPS[@]}"; do
  for local in 0 1; do
    echo "=== exp$e  bg_local=$local ($([[ $local == 0 ]] && echo 'cross-die probe' || echo 'own-die control')) ==="
    NVHBI_BG_LOCAL="$local" ./"$PROG" "$e" 2>&1 | tee -a "$LOG_OUT"
    echo
  done
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
Read the CALIBRATION block first: it names the FAR die and warns if NVLink is
not clearly die-attached. If it warns, exp2 vs exp3 may not isolate NV-HBI.

Plot 2a  x = peer_GBps (NVLink load)   y = bg_GBps (remaining bisection)
         series: exp2/bg_local=0 vs exp3/bg_local=0, at the max bg_sms.
         exp2 should fall as peer competes for the bisection; exp3 stays flat.
         (With NV-HBI >> NVLink the fall only appears once bg saturates NV-HBI,
          so use the largest bg_sms.)

Plot 2c  x = bg_sms   y = peer_GBps
         series: exp2 vs exp3. exp2's peer throughput drops as the background
         saturates the bisection; exp3's does not.

Control  bg_local=1 repeats both with a background that never crosses the
         fabric. Any effect common to bg_local=0 and 1 is SM/L2 contention,
         not bisection -- subtract it.
EOF

echo
echo "NVLink counters (independent cross-check of the peer side):"
nvidia-smi nvlink -gt d 2>/dev/null | head -20 || echo "  (nvidia-smi nvlink unavailable)"
