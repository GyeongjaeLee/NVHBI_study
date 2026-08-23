#!/bin/bash
# run_dualdir.sh -- ONE fabric direction driven by two payload sources.
#
#   die A SMs write into die B          payload A -> B
#   die B SMs read  from die A          payload A -> B as well
#
# Writes dualdir.csv (crossing reads) and, with NVHBI_R_LOCAL=1,
# dualdir_local.csv -- the control where the readers stay on their own die and
# nothing crosses. Comparing the two is how you tell whether reads and writes
# meet at the die boundary or somewhere on the destination die.
#
# Usage:
#   ./run_dualdir.sh                  # crossing reads  -> dualdir.csv
#   NVHBI_R_LOCAL=1 ./run_dualdir.sh  # local reads     -> dualdir_local.csv
#
# Env: CSV_OUT, GPU_ID + every NVHBI_* knob nvhbi_dualdir.cu reads.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./nvhbi_lib.sh

GPU_ID="${GPU_ID:-0}"
PROG="nvhbi_dualdir"
PARTITION="${1:-0}"
R_LOCAL="${NVHBI_R_LOCAL:-0}"
CSV_OUT="${CSV_OUT:-$([[ "$R_LOCAL" == 1 ]] && echo dualdir_local.csv || echo dualdir.csv)}"

CFG_HEADER="w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps,dieB_GBps"

nvhbi_detect_gpu "$GPU_ID"
nvhbi_build "$PROG"

nvhbi_clock_snapshot "before" "$GPU_ID"
{
  echo "$CFG_HEADER"
  NVHBI_R_LOCAL="$R_LOCAL" CUDA_VISIBLE_DEVICES="$GPU_ID" ./"$PROG" "$PARTITION" \
    | awk '/^CFG,/ { sub(/^CFG,/, ""); print }'
} > "$CSV_OUT"
nvhbi_clock_snapshot "after " "$GPU_ID"

rows=$(( $(wc -l < "$CSV_OUT" | tr -d ' ') - 1 ))
echo "Done. Wrote $CSV_OUT ($rows rows)"

cat <<'EOF'

How to read it
--------------
write_GBps  cross-die stores, the only column whose bytes are unambiguous
            (a 4B store ships one 32B sector).
read_GBps   the read payload on the same direction.
total_GBps  write + read when the reads cross; write alone when they do not,
            because then only the writes are on the fabric.
dieB_GBps   everything arriving at the destination die, crossing or not.

The comparison that matters: run it twice, once with crossing reads and once
with NVHBI_R_LOCAL=1. If write_GBps collapses under crossing reads and holds
under local reads at the same or higher read rate, the two payloads are meeting
at the die boundary and not on the destination die.
EOF
