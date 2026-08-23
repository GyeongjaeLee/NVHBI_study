#!/bin/bash
# nvlink_check.sh -- is peer_GBps the number of bytes that actually cross NVLink?
#
# exp2/exp3 report the peer stream as issued_stores x 32B, an identity that was
# calibrated for ON-DIE SM stores into a remote L2 and never for stores arriving
# over NVLink. 633 GB/s of "sectors" is 19.8 G stores/s, and that is consistent
# with two very different pictures:
#
#   (a) each store carries a full 32B sector       -> 633 GB/s on the wire,
#                                                     70% of the 900 GB/s that
#                                                     18 NVLink-5 links give in
#                                                     one direction. Normal.
#   (b) each store carries a 4B payload and the
#       link is limited by PACKET rate             -> 79 GB/s on the wire, and
#                                                     the peer's load on NV-HBI
#                                                     is 8x smaller than exp2
#                                                     reports -- which would
#                                                     make "no interference"
#                                                     trivially true.
#
# The hardware link counters settle it, but only with a known interval and with
# the peer running for the whole of it. A bare pair of `nvidia-smi nvlink -gt d`
# calls around a sweep cannot: the sweep leaves the link idle during every
# flush, warm-up and settle.
#
# So: run ONE peer-only point, long, and bracket it.
#
#   ./nvlink_check.sh                  # 10 s, peer only, block size 128
#   SECS=20 BSIZE=64 ./nvlink_check.sh
#
# Compare the two numbers it prints. Equal (within a few %) => (a), and the 32B
# accounting is sound on the NVLink path too. Wire ~= reported/8 => (b).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SECS="${SECS:-10}"
BSIZE="${BSIZE:-128}"
EXP="${EXP:-2}"

# Sum of Data Rx over every link of GPU 0, in KiB. Tx is zero for a pure peer
# write, and the two GPUs report the same counter from both ends, so one
# direction of one GPU is the whole story.
rx_kib() {
  nvidia-smi nvlink -gt d -i 0 \
    | awk '/Data Rx/ { gsub(/,/,"",$(NF-1)); s += $(NF-1) } END { printf "%.0f", s }'
}

echo "peer-only, block size $BSIZE, one $((SECS*1000)) ms window"
echo

A=$(rx_kib); TA=$(date +%s.%N)
NVHBI_BG_SMS_LIST=0 \
NVHBI_PEER_BLOCK_SIZES="$BSIZE" \
NVHBI_REPEAT=1 \
NVHBI_WINDOW_MS=$((SECS*1000)) \
  ./nvhbi_exp23_peer "$EXP" 2>&1 | tee /tmp/nvlink_check.log | grep -E '^CFG,|peer WRITE|NEAR die|FAR  die'
TB=$(date +%s.%N); B=$(rx_kib)

DT=$(echo "$TB - $TA" | bc -l)
WIRE=$(echo "($B - $A) * 1024 / $DT / 1000000000" | bc -l)

# peer_GBps x peer_ms from the CFG row, i.e. bytes the kernel THINKS it sent.
REPORTED=$(awk -F, '/^CFG,/ { printf "%.1f", $9 }' /tmp/nvlink_check.log | tail -1)
PEER_MS=$(awk -F, '/^CFG,/ { printf "%.1f", $8 }' /tmp/nvlink_check.log | tail -1)

echo
echo "----------------------------------------------------------"
printf "wall interval                : %.2f s\n" "$DT"
printf "NVLink Data Rx delta         : %.1f GB\n" \
       "$(echo "($B - $A) * 1024 / 1000000000" | bc -l)"
printf "  => wire bandwidth          : %.1f GB/s   (over the whole interval)\n" "$WIRE"
printf "kernel-reported peer_GBps    : %s GB/s over %s ms\n" "$REPORTED" "$PEER_MS"
echo
echo "The kernel was only storing for peer_ms of the wall interval, so scale:"
printf "  reported bytes             : %.1f GB\n" \
       "$(echo "$REPORTED * $PEER_MS / 1000" | bc -l)"
echo
echo "reported bytes ~= NVLink delta  -> a 4B peer store really does put a 32B"
echo "                                   sector on the wire; exp2's accounting"
echo "                                   holds on the NVLink path."
echo "NVLink delta ~= reported / 8    -> only the 4B payload crosses. The peer's"
echo "                                   real load on NV-HBI is 8x smaller than"
echo "                                   exp2 reports, and every 'no"
echo "                                   interference' row needs reinterpreting."
