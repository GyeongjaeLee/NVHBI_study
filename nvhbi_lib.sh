#!/bin/bash
# nvhbi_lib.sh -- shared helpers for the run_exp*.sh scripts. Source, don't run.

nvhbi_detect_gpu() {
  local gid="${1:-0}"
  NVHBI_GPU_NAME=$(nvidia-smi --query-gpu=name,index --format=csv,noheader \
                   | awk -F, -v g="$gid" 'int($2)==g {print $1; exit}')
  if [[ -z "$NVHBI_GPU_NAME" ]]; then
    echo "Failed to read GPU name for gpu_id=$gid" >&2
    return 1
  fi
  shopt -s nocasematch
  if [[ "$NVHBI_GPU_NAME" == *"B300"* || "$NVHBI_GPU_NAME" == *"GB300"* ]]; then
    # B300/Blackwell-Ultra: compute capability not assumed here. Default to 100
    # and let ARCH= override if nvcc rejects it (e.g. ARCH=103).
    NVHBI_ARCH="${ARCH:-100}"
    echo "NOTE: B300 detected; building for sm_$NVHBI_ARCH (override with ARCH= if nvcc errors)." >&2
  elif [[ "$NVHBI_GPU_NAME" == *"B200"* || "$NVHBI_GPU_NAME" == *"GB200"* ]]; then
    NVHBI_ARCH="100"
  elif [[ "$NVHBI_GPU_NAME" == *"H100"* ]]; then
    NVHBI_ARCH="90"
    echo "WARNING: H100 is a single die -- there is no NV-HBI bisection to measure." >&2
  elif [[ "$NVHBI_GPU_NAME" == *"A100"* ]]; then
    NVHBI_ARCH="80"
    echo "WARNING: A100 is a single die -- there is no NV-HBI bisection to measure." >&2
  else
    NVHBI_ARCH="${ARCH:-100}"
    echo "WARNING: unrecognized GPU '$NVHBI_GPU_NAME', assuming sm_$NVHBI_ARCH." >&2
  fi
  shopt -u nocasematch
  echo "GPU $gid: $NVHBI_GPU_NAME (sm_$NVHBI_ARCH)"
}

# Snapshot the clock/power/thermal state. Identical work measured in two
# sessions differed by 1.56x on B200, so the machine state has to be recorded
# alongside every bandwidth number or the absolute values are not comparable.
nvhbi_clock_snapshot() {
  local tag="$1" gid="${2:-0}"
  local q="clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu"
  local row
  row=$(nvidia-smi -i "$gid" --query-gpu="$q" --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "$row" ]]; then echo "[$tag] clocks: unavailable"; return; fi
  IFS=',' read -r sm smmax mem pw pwlim temp util <<<"$row"
  sm=${sm// /}; smmax=${smmax// /}; util=${util// /}
  echo "[$tag] sm_clock=${sm}/${smmax} MHz  mem=${mem// /} MHz  power=${pw// /}/${pwlim// /} W  temp=${temp// /}C  util=${util}%"
  # An idle GPU parks at its minimum clock (throttle reason 0x1 = GpuIdle), so a
  # snapshot taken before any kernel runs always looks throttled. Only warn when
  # something was actually running.
  if [[ "$smmax" =~ ^[0-9]+$ && "$sm" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
     && (( util > 5 )) && (( sm * 100 < smmax * 80 )); then
    echo "[$tag] WARNING: SM clock is under 80% of max -- bandwidth numbers from this" >&2
    echo "[$tag]          run are not comparable with runs at full clock." >&2
    nvidia-smi -i "$gid" --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>/dev/null \
      | sed "s/^/[$tag] throttle reasons: /" || true
  fi
  # Anything else on the GPU invalidates the measurement outright.
  local others
  others=$(nvidia-smi -i "$gid" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  [[ "$others" -gt 1 ]] && echo "[$tag] WARNING: $others compute processes on GPU $gid" >&2
  return 0
}

nvhbi_build() {
  local target="$1"
  make -f Makefile.nvhbi ARCH="$NVHBI_ARCH" "$target"
}

# nvhbi_cfg_rows <file> <expected_field_count>
# Program config rows look like "CFG,<fields>"; the "# CFG,..." header is skipped.
nvhbi_cfg_rows() {
  awk -v want="$2" '
      index($0, "# CFG,") { next }
      {
          i = index($0, "CFG,")
          if (i > 0) {
              row = substr($0, i + 4)
              if (split(row, f, ",") == want) print row
          }
      }
  ' "$1"
}
