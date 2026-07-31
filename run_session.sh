#!/bin/bash
# run_session.sh -- run every experiment against ONE GPU allocation.
#
# Why this exists: the same configuration measured 2007 GB/s on one RunPod
# instance and 2676 on another -- a 1.33x spread. Within a single session the
# two independent harnesses agree to 2.7% (exp1's slope method vs exp2's store
# counter), so numbers are only comparable when they come from the same
# allocation. Mixing sessions is what made the earlier exp1-vs-exp2 gap look
# like a code bug when it was not.
#
# Everything lands in one directory with a manifest recording the GPU, clocks,
# SM split and detected far die, so a stray CSV can never be attributed to the
# wrong session.
#
#   ./run_session.sh              # gate, then exp1 + exp2 + exp3
#   ./run_session.sh --gate       # just the 60-second instance check
#   ./run_session.sh --with-nccl  # also exp4 (needs NCCL)
#   OUTDIR=mydir ./run_session.sh
#
# Env: GATE_MIN_GBPS  warn below this at max SMs (default 4000)
#      plus every NVHBI_* knob the binaries read.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

GATE_ONLY=0
WITH_NCCL=0
for a in "$@"; do
  case "$a" in
    --gate) GATE_ONLY=1 ;;
    --with-nccl) WITH_NCCL=1 ;;
    *) echo "usage: $0 [--gate] [--with-nccl]" >&2; exit 1 ;;
  esac
done

GPU_ID="${GPU_ID:-0}"
SESSION="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-session_$SESSION}"
GATE_MIN_GBPS="${GATE_MIN_GBPS:-4000}"
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/manifest.txt"

exec > >(tee "$OUTDIR/run.log") 2>&1

echo "=== session $SESSION -> $OUTDIR ==="
nvhbi_detect_gpu "$GPU_ID"
nvhbi_build all

{
  echo "session_id: $SESSION"
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo
  echo "[gpus]"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.max.sm,clocks.max.mem,power.limit \
             --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
  echo
  echo "[topology]"
  nvidia-smi topo -m 2>/dev/null | head -12 || true
} > "$MANIFEST"

nvhbi_clock_snapshot "session-start" "$GPU_ID" | tee -a "$MANIFEST"
echo

# ---------------------------------------------------------------- gate
# One point at max SMs. Instance-to-instance spread is 1.33x, so check what this
# allocation is capable of before spending time on the full sweep -- and record
# it either way. Do NOT silently re-roll instances until the number looks good:
# if you keep this one, keep its number.
echo "--- gate: cross-die bandwidth at max SMs ---"
GATE_CSV="$OUTDIR/gate.csv"
NVHBI_SMS=999 NVHBI_BLOCK_SIZES=64 NVHBI_OWN_DIE=0,1 \
  CSV_OUT="$GATE_CSV" ./run_exp1_bisection.sh 2 > "$OUTDIR/gate.log" 2>&1 || {
    echo "gate FAILED -- see $OUTDIR/gate.log" >&2; exit 1; }

awk -F',' 'NR>1 {printf "  wp=%s own_die=%s  %s SMs  slope %8.1f  counted %8.1f  ratio %s\n",$1,$2,$3,$12,$13,$14}' "$GATE_CSV"
if [[ $(wc -l < "$GATE_CSV") -le 1 ]]; then
  echo "  ERROR: gate produced no rows. NVHBI_SMS may name a count this GPU does not have." >&2
  exit 1
fi
CROSS_MAX=$(awk -F',' 'NR>1 && $2==0 {if ($12+0 > m) m=$12+0} END {printf "%.0f", m}' "$GATE_CSV")
OWN_MAX=$(awk -F',' 'NR>1 && $2==1 {if ($12+0 > m) m=$12+0} END {printf "%.0f", m}' "$GATE_CSV")
echo "  cross-die max: $CROSS_MAX GB/s   own-die max: $OWN_MAX GB/s"
if [[ -n "$CROSS_MAX" && -n "$OWN_MAX" && "$CROSS_MAX" -gt 0 ]]; then
  awk -v c="$CROSS_MAX" -v o="$OWN_MAX" 'BEGIN{printf "  own/cross ratio: %.3f  (stable at ~1.29-1.33 across sessions;\n                    this ratio is the saturation evidence and does not\n                    depend on how fast the instance happens to be)\n", o/c}'
fi
if [[ "$CROSS_MAX" -lt "$GATE_MIN_GBPS" ]]; then
  echo
  echo "  NOTE: $CROSS_MAX GB/s is below GATE_MIN_GBPS=$GATE_MIN_GBPS. Fast instances have"
  echo "        reached ~4750. Everything measured here will be consistent with this"
  echo "        number, so the session is still internally valid -- but if you need a"
  echo "        headline absolute figure, this allocation will understate it."
fi
echo
[[ "$GATE_ONLY" == "1" ]] && { echo "gate only; stopping."; exit 0; }

# ---------------------------------------------------------------- exp1
echo "--- exp1: bisection vs #SMs (cross + own die, both partitions) ---"
CSV_OUT="$OUTDIR/exp1.csv" ./run_exp1_bisection.sh 2 > "$OUTDIR/exp1.log" 2>&1
echo "  -> $OUTDIR/exp1.csv ($(( $(wc -l < "$OUTDIR/exp1.csv") - 1 )) rows)"
nvhbi_clock_snapshot "after-exp1" "$GPU_ID" | tee -a "$MANIFEST"

# ---------------------------------------------------------------- exp2/3
EXP23_HEADER="exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl,peer_bsize"
ndev=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$ndev" -lt 2 ]]; then
  echo "--- exp2/exp3 skipped: needs 2 GPUs, found $ndev ---"
else
  for e in 2 3; do
    for local in 0 1; do
      name="exp${e}_bglocal${local}"
      echo "--- $name ---"
      NVHBI_BG_LOCAL="$local" ./nvhbi_exp23_peer "$e" > "$OUTDIR/$name.log" 2>&1
      { echo "$EXP23_HEADER"; nvhbi_cfg_rows "$OUTDIR/$name.log" 14; } > "$OUTDIR/$name.csv"
      echo "  -> $OUTDIR/$name.csv ($(( $(wc -l < "$OUTDIR/$name.csv") - 1 )) rows)"
    done
  done
  # The calibration is identical for all four runs; keep one copy where it is
  # easy to find, since far_die changes between sessions.
  sed -n '/=== CALIBRATION/,/====$/p' "$OUTDIR/exp2_bglocal0.log" >> "$MANIFEST" 2>/dev/null || true
  grep -m1 "FAR  die" "$OUTDIR/exp2_bglocal0.log" >> "$MANIFEST" 2>/dev/null || true
fi
nvhbi_clock_snapshot "after-exp23" "$GPU_ID" | tee -a "$MANIFEST"

# ------------------------------------------------------- dualdir (the ceiling)
# THIS IS THE ROW EXP2 HAS TO BE READ AGAINST, and it has to come from the same
# allocation. exp2's background tops out around 3.4-3.5 TB/s cross-die, but that
# number alone cannot say whether the limit is the NV-HBI link or the 70 writing
# SMs on the source die. dualdir adds a SECOND payload source on the same
# direction (die-B SMs reading die-A memory, so the payload also travels A->B)
# using SMs the writers do not have. If the total still stops at the write-alone
# figure, the direction really is link-limited; if it climbs, 3.4 TB/s was a
# source-side limit and NV-HBI was never saturated.
#
# Read it next to exp2's crossing_GBps: exp2 reports background + peer summing
# to ~4.1 TB/s on that same direction. Those two cannot both be right unless the
# peer traffic is not actually crossing.
echo "--- dualdir: one direction, two payload sources (the NV-HBI ceiling) ---"
DUAL_HEADER="w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps"
: > "$OUTDIR/dualdir.log"
for rloc in 0 1; do
  NVHBI_R_LOCAL="$rloc" ./nvhbi_dualdir 0 >> "$OUTDIR/dualdir.log" 2>&1 || true
done
{ echo "$DUAL_HEADER"; nvhbi_cfg_rows "$OUTDIR/dualdir.log" 7; } > "$OUTDIR/dualdir.csv"
echo "  -> $OUTDIR/dualdir.csv ($(( $(wc -l < "$OUTDIR/dualdir.csv") - 1 )) rows)"

# --------------------------------------------------- peerl2 (L2 or HBM?)
if [[ "$ndev" -ge 2 ]]; then
  echo "--- peerl2: is GPU1 served from GPU0's L2 or from GPU0's HBM? ---"
  PL2_HEADER="kind,die,is_far,state,cv,corun,rep,peer_cyc,local_cyc,peer_GBps,co_GBps,chunks"
  ./nvhbi_peerl2 > "$OUTDIR/peerl2.log" 2>&1 || true
  { echo "$PL2_HEADER"; nvhbi_cfg_rows "$OUTDIR/peerl2.log" 12; } > "$OUTDIR/peerl2.csv"
  echo "  -> $OUTDIR/peerl2.csv ($(( $(wc -l < "$OUTDIR/peerl2.csv") - 1 )) rows)"
fi
nvhbi_clock_snapshot "after-dual-peerl2" "$GPU_ID" | tee -a "$MANIFEST"

# ---------------------------------------------------------------- exp4
if [[ "$WITH_NCCL" == "1" && "$ndev" -ge 2 ]]; then
  echo "--- exp4: NCCL all-to-all under contention ---"
  nvhbi_find_nccl || true
  if nvhbi_build nccl; then
    for local in 0 1; do
      name="exp4_bglocal${local}"
      NVHBI_BG_LOCAL="$local" ./nvhbi_exp4_nccl > "$OUTDIR/$name.log" 2>&1
      { echo "exp,bg_local,bg_sms,msg_bytes,rep,nccl_ms,algbw_GBps,busbw_GBps,bg_GBps";
        nvhbi_cfg_rows "$OUTDIR/$name.log" 9; } > "$OUTDIR/$name.csv"
      echo "  -> $OUTDIR/$name.csv"
    done
  else
    echo "  NCCL build failed; set NCCL_HOME. Skipping exp4." >&2
  fi
fi

nvhbi_clock_snapshot "session-end" "$GPU_ID" | tee -a "$MANIFEST"
echo
echo "=== done: $OUTDIR ==="
ls -1 "$OUTDIR"/*.csv 2>/dev/null | sed 's/^/  /'
cat <<EOF

All CSVs above come from ONE allocation and are directly comparable. Do not pool
them with CSVs from another session -- the same config varies 1.33x between
instances, and far_die flips too.

manifest.txt holds the GPU identity, the clock snapshots taken between
experiments, and the calibration block naming this session's far die.
EOF
