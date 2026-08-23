#!/bin/bash
# run_session.sh -- run every experiment against ONE GPU allocation.
#
# Numbers are only comparable inside a session: a cloud allocation hands out a
# different physical B200 each time, with a different SM split between the dies
# and a different clock/thermal state. Everything therefore lands in one
# directory with a manifest recording the GPU, its clocks, the SM split and the
# detected NVLink-attached die.
#
# What it produces (all CSV, ready for plot_nvhbi.py):
#
#   route_probe.log        validity checks: warm coverage, block->SM slots,
#                          NVLink attach die, where a peer write lands
#   exp1.csv               cross-die vs own-die write bandwidth vs #SMs
#   dualdir.csv            one direction, writes + crossing reads
#   dualdir_local.csv      same, reads kept local: the control
#   exp2_bglocal0.csv      peer -> FAR die, background crossing
#   exp2_bglocal1.csv        "        "     background own-die (control)
#   exp3_bglocal0.csv      peer -> NEAR die (control), background crossing
#   exp3_bglocal1.csv        "        "     background own-die
#
#   ./run_session.sh              # everything
#   ./run_session.sh --quick      # skip the exp2/exp3 reader sweep
#   OUTDIR=mydir ./run_session.sh
#
# Env: BG_R_SMS   reader sweep for exp2/exp3 (default "0,8,16,32,72")
#      BUF_MULT   allocation as a multiple of L2 for the read sweeps (64)
#      plus every NVHBI_* knob the binaries read.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

QUICK=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    *) echo "usage: $0 [--quick]" >&2; exit 1 ;;
  esac
done

GPU_ID="${GPU_ID:-0}"
SESSION="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-session_$SESSION}"
BG_R_SMS="${BG_R_SMS:-0,8,16,32,72}"
BUF_MULT="${BUF_MULT:-64}"
[[ "$QUICK" == 1 ]] && BG_R_SMS="0"
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/manifest.txt"

exec > >(tee "$OUTDIR/run.log") 2>&1

echo "=== session $SESSION -> $OUTDIR ==="
nvhbi_detect_gpu "$GPU_ID"
nvhbi_build all
make -f Makefile.nvhbi probes >/dev/null

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

# Extract "CFG," rows into a CSV with the header the program printed.
cfg_to_csv() {   # cfg_to_csv <log> <csv> <fallback-header>
  local log="$1" csv="$2" hdr="$3"
  local prog_hdr
  prog_hdr="$(grep -m1 '^# CFG,' "$log" 2>/dev/null | sed 's/^# CFG,//')" || true
  { echo "${prog_hdr:-$hdr}"
    awk '/^CFG,/ { sub(/^CFG,/, ""); print }' "$log"
  } > "$csv"
  echo "  -> $csv ($(( $(wc -l < "$csv") - 1 )) rows)"
}

# ---------------------------------------------------------------- validity
echo; echo "=== route probe (validity checks) ==="
./nvhbi_route_probe > "$OUTDIR/route_probe.log" 2>&1 || true
grep -E 'coverage|distinct values|VERDICT|NEAR die|mismatch|H_internal' \
     "$OUTDIR/route_probe.log" || true
{ echo; echo "[route probe]"; grep -E 'NEAR die|FAR  die|H_internal|VERDICT' \
     "$OUTDIR/route_probe.log" || true; } >> "$MANIFEST"

# ---------------------------------------------------------------- exp1
echo; echo "=== exp1: cross-die vs own-die write, swept over #SMs ==="
nvhbi_clock_snapshot "exp1 before" "$GPU_ID"
NVHBI_SMS="${NVHBI_SMS:-}" ./nvhbi_exp1_bisection 2 > "$OUTDIR/exp1.log" 2>&1
cfg_to_csv "$OUTDIR/exp1.log" "$OUTDIR/exp1.csv" \
  "writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,footprint_MB,sectors_lo,sectors_hi,ms_lo,ms_hi,naive_GBps,slope_GBps,counted_GBps,count_ratio,overhead_ms,eff_GHz,span_ratio,sampled_GBps,sample_ms"

# ---------------------------------------------------------------- dualdir
for rloc in 0 1; do
  name=$([[ $rloc == 0 ]] && echo dualdir || echo dualdir_local)
  echo; echo "=== $name: one direction, writes + reads (r_local=$rloc) ==="
  NVHBI_BUF_MULT="$BUF_MULT" NVHBI_R_LOCAL="$rloc" \
  NVHBI_W_SMS="${W_SMS:-0,999}" NVHBI_R_SMS="${R_SMS:-0,8,16,32,72}" \
    ./nvhbi_dualdir 0 > "$OUTDIR/$name.log" 2>&1
  cfg_to_csv "$OUTDIR/$name.log" "$OUTDIR/$name.csv" \
    "w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps,dieB_GBps"
done

# ---------------------------------------------------------------- exp2/exp3
for e in 2 3; do
  for loc in 0 1; do
    name="exp${e}_bglocal${loc}"
    echo; echo "=== $name ==="
    NVHBI_BUF_MULT="$BUF_MULT" NVHBI_BG_LOCAL="$loc" NVHBI_BG_R_SMS="$BG_R_SMS" \
    NVHBI_BG_SMS_LIST="${BG_SMS_LIST:-0,16,32,64,999}" \
      ./nvhbi_exp23_peer "$e" > "$OUTDIR/$name.log" 2>&1
    cfg_to_csv "$OUTDIR/$name.log" "$OUTDIR/$name.csv" \
      "exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl,peer_bsize,bg_rd_GBps,bg_r_sms"
  done
done

nvhbi_clock_snapshot "session end" "$GPU_ID"
{ echo; nvhbi_clock_snapshot "session-end" "$GPU_ID"; } >> "$MANIFEST" 2>/dev/null || true

echo
echo "=== done: $OUTDIR ==="
ls -1 "$OUTDIR"/*.csv
echo
echo "Plot with:  python3 plot_nvhbi.py $OUTDIR"
