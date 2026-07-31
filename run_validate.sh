#!/bin/bash
# run_validate.sh -- TEMPORARY: ncu validation of the store-counter methodology.
#
# Needs GPU performance-counter access (unavailable on the B200 containers; the
# point of running this on B300). Launches nvhbi_stress_write once per config and
# compares ncu's fabric/write sector counts against the deterministic analytic
# store count.
#
#   ./run_validate.sh                 # cross-die, ncu
#   ./run_validate.sh 1               # partition-1 writers
#   USE_NCU=0 ./run_validate.sh       # counter-arithmetic check only (no ncu)
#   NVHBI_OWN_DIE=0,1 ./run_validate.sh
#
# Read the result:
#   count_ratio  (non-ncu)  counted/analytic. Must be 1.0000 or the counting
#                           code itself is wrong -- fix that before believing ncu.
#   write_ratio  ncu tex_op_write / analytic. ~1 => SM issued every store.
#   fabric_ratio ncu ltcfabric   / analytic. ~1 => every store crossed as a 32B
#                sector; << 1 => absorption, and the methodology overstates
#                bandwidth by 1/fabric_ratio. Compare L=1 (same-address hammer)
#                against L=8 (more distinct sectors): a gap that only shows at
#                L=1 is write-combining of repeated stores.
#
# Env: GPU_ID ARCH USE_NCU CSV_OUT NCU NCU_SUDO
#      FABRIC_METRIC WRITE_METRIC + every NVHBI_* knob the binary reads.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

GPU_ID="${GPU_ID:-0}"
USE_NCU="${USE_NCU:-1}"
PROG="nvhbi_validate_counter"
KERNEL="${KERNEL:-nvhbi_stress_write}"
PARTITION="${1:-0}"
CSV_OUT="${CSV_OUT:-validate_counter.csv}"
NCU="${NCU:-ncu}"
NCU_SUDO="${NCU_SUDO:-0}"

# Write-only traffic, so the plain (read+write) fabric metric equals the write
# sectors. If your ncu build exposes the op_write variants, they are cleaner:
#   FABRIC_METRIC=lts__t_sectors_srcunit_ltcfabric_op_write.sum
FABRIC_METRIC="${FABRIC_METRIC:-lts__t_sectors_srcunit_ltcfabric.sum}"
WRITE_METRIC="${WRITE_METRIC:-lts__t_sectors_srcunit_tex_op_write.sum}"

CFG_FIELDS=11   # writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,
                # lines_mult,iteration,analytic_sectors,counted_sectors,count_ratio,ms
CFG_HEADER="writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,lines_mult,iteration,analytic_sectors,counted_sectors,count_ratio,ms"

# ncu prints absolute counts with a magnitude prefix: "sector", "Ksector",
# "Msector", "Gsector". Return one integer per profiled launch, in order.
ncu_count() {
  awk -v metric="$1" '
      index($0, metric) {
          unit = ""; val = "";
          for (i = 1; i <= NF; i++) {
              if ($i ~ /^[KMGT]?sectors?$/) { unit = $i; continue }
              tok = $i; gsub(/,/, "", tok);
              if (tok ~ /^[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$/) val = tok;
          }
          if (val == "") next;
          m = 1; p = substr(unit, 1, 1);
          if      (p == "K") m = 1e3;  else if (p == "M") m = 1e6;
          else if (p == "G") m = 1e9;  else if (p == "T") m = 1e12;
          printf "%.0f\n", val * m;
      }
  ' "$2"
}

nvhbi_detect_gpu "$GPU_ID"
nvhbi_build "$PROG"

run_prog() { CUDA_VISIBLE_DEVICES="$GPU_ID" ./"$PROG" "$PARTITION"; }

if [[ "$USE_NCU" != "1" ]]; then
  echo "counter-arithmetic check only (no profiler)"
  { echo "$CFG_HEADER"; run_prog | awk '/^CFG,/ { sub(/^CFG,/,""); print }'; } > "$CSV_OUT"
  echo "Wrote $CSV_OUT"
  echo "count_ratio must be 1.0000 on every row:"
  awk -F',' 'NR>1 {printf "  sm=%s bs=%s L=%s  count_ratio=%s\n",$3,$5,$6,$10}' "$CSV_OUT"
  exit 0
fi

NCU_CMD=("$NCU"); [[ "$NCU_SUDO" == "1" ]] && NCU_CMD=(sudo "$NCU")
tmp_out="$(mktemp)"; tmp_cfg="$(mktemp)"; tmp_fab="$(mktemp)"; tmp_wr="$(mktemp)"
trap 'rm -f "$tmp_out" "$tmp_cfg" "$tmp_fab" "$tmp_wr"' EXIT

echo "Profiling '$KERNEL'"
echo "  fabric: $FABRIC_METRIC"
echo "  write : $WRITE_METRIC"

# --cache-control none: keep the warmed remote lines resident between replay
# passes, or (write-no-allocate) every store misses and the test becomes an HBM
# test. --clock-control none: do not let ncu renormalize clocks.
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" "${NCU_CMD[@]}" \
    --replay-mode kernel --cache-control none --clock-control none \
    --metrics "$FABRIC_METRIC,$WRITE_METRIC" \
    --kernel-name "$KERNEL" \
    ./"$PROG" "$PARTITION" > "$tmp_out" 2>&1
rc=$?
set -e
if grep -q "ERR_NVGPUCTRPERM" "$tmp_out"; then
  echo "ERROR: ncu lacks GPU counter permission (ERR_NVGPUCTRPERM) on this host too." >&2
  echo "       Run as root or set the NVreg_RestrictProfilingToAdminUsers=0 module option." >&2
  exit 1
fi
if [[ $rc -ne 0 ]]; then echo "ncu exited $rc:" >&2; cat "$tmp_out" >&2; exit $rc; fi

awk '
    index($0, "# CFG,") { next }
    { i = index($0, "CFG,"); if (i > 0) { row = substr($0, i+4);
        if (split(row, f, ",") == '"$CFG_FIELDS"') print row } }
' "$tmp_out" > "$tmp_cfg"
ncu_count "$FABRIC_METRIC" "$tmp_out" > "$tmp_fab"
ncu_count "$WRITE_METRIC"  "$tmp_out" > "$tmp_wr"

n_cfg=$(wc -l < "$tmp_cfg" | tr -d ' ')
n_fab=$(wc -l < "$tmp_fab" | tr -d ' ')
n_wr=$(wc -l < "$tmp_wr" | tr -d ' ')
echo "rows: cfg=$n_cfg fabric=$n_fab write=$n_wr"
if [[ "$n_cfg" -eq 0 || "$n_fab" -eq 0 ]]; then
  echo "ERROR: no CFG rows or no fabric samples." >&2
  echo "  list metrics: $NCU --query-metrics | grep -E 'ltcfabric|tex_op_write'" >&2
  cat "$tmp_out" >&2; exit 1
fi
if [[ "$n_cfg" -ne "$n_fab" ]]; then
  echo "ERROR: cfg rows ($n_cfg) != fabric samples ($n_fab); kernel-name may match extra launches." >&2
  exit 1
fi
# write metric is optional (variant may not exist on this ncu)
HAVE_WR=0; [[ "$n_wr" -eq "$n_cfg" ]] && HAVE_WR=1
[[ "$HAVE_WR" -eq 0 ]] && echo "note: write metric absent/mismatched; reporting fabric_ratio only." >&2

{
  if [[ "$HAVE_WR" -eq 1 ]]; then
    echo "$CFG_HEADER,ncu_fabric_sectors,fabric_ratio,ncu_write_sectors,write_ratio"
    paste -d',' "$tmp_cfg" "$tmp_fab" "$tmp_wr" | awk -F',' -v OFS=',' '{
        a=$8+0; fab=$12+0; wr=$13+0;
        fr=(a>0?sprintf("%.4f",fab/a):"0"); wrr=(a>0?sprintf("%.4f",wr/a):"0");
        print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, fab, fr, wr, wrr;
    }'
  else
    echo "$CFG_HEADER,ncu_fabric_sectors,fabric_ratio"
    paste -d',' "$tmp_cfg" "$tmp_fab" | awk -F',' -v OFS=',' '{
        a=$8+0; fab=$12+0;
        print $0, (a>0?sprintf("%.4f",fab/a):"0");
    }'
  fi
} > "$CSV_OUT"

echo "Wrote $CSV_OUT"
echo
echo "Interpretation:"
echo "  fabric_ratio ~1.0  => every 4B store crossed as one 32B sector; the counter"
echo "                        methodology is sound and reports real bisection traffic."
echo "  fabric_ratio <1.0  => stores are being absorbed; multiply all counter-based"
echo "                        bandwidths (bg_GBps, peer_GBps) by this to correct them."
[[ "$HAVE_WR" -eq 1 ]] && \
echo "  write_ratio vs fabric_ratio  => if write~1 but fabric<1, the merge is at the"
echo "                        fabric/L2; if both<1, the SM/compiler dropped stores."
