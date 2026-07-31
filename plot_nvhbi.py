#!/usr/bin/env python3
"""plot_nvhbi.py -- figures for one NV-HBI session directory.

Reads a session produced by run_session.sh (or the raw .log files) and writes:

  exp1_remote_l2_write.png   x = #SMs, y = GB/s, one line per block_size
  exp1_local_l2_write.png    same, for own-die writes
  exp2_bglocal0.png          x = NVLink load, y = bg GB/s and peer GB/s,
  exp2_bglocal1.png          one line per background SM count
  exp3_bglocal0.png
  exp3_bglocal1.png

Prefers <name>.csv; falls back to extracting the "CFG," rows from <name>.log,
so it works on a directory that only has logs.

Usage:
    python3 plot_nvhbi.py SESSION_DIR [-o OUTDIR]
"""

import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# exp1 gained counted_GBps/count_ratio partway through, so always address
# columns by name and never by position.
EXP1_HEADER_HINT = "writer_partition,own_die,num_active_sm"
EXP23_HEADER = ("exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,"
                "peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz")
EXP4_HEADER = "exp,bg_local,bg_sms,msg_bytes,rep,nccl_ms,algbw_GBps,busbw_GBps,bg_GBps"
DUAL_HEADER = "w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps"


def load_rows(session_dir, stem, header_if_log):
    """Return list[dict] from <stem>.csv, else from the CFG rows of <stem>.log."""
    csv_path = os.path.join(session_dir, stem + ".csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            return rows

    log_path = os.path.join(session_dir, stem + ".log")
    if not os.path.exists(log_path):
        return []

    body = []
    header = header_if_log
    for line in open(log_path):
        s = line.strip()
        if s.startswith("# CFG,"):
            # the program's own header line, authoritative if present
            header = s[len("# CFG,"):]
        elif s.startswith("CFG,"):
            body.append(s[len("CFG,"):])
    if not body:
        return []
    return list(csv.DictReader(io.StringIO(header + "\n" + "\n".join(body))))


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def i(row, key):
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return -1


def plot_exp1(rows, outdir):
    """One figure per die target: x = #SMs, y = GB/s, line per block_size."""
    if not rows:
        print("exp1: no rows, skipping")
        return

    ycol = "slope_GBps" if "slope_GBps" in rows[0] else "counted_GBps"

    for own_die, fname, title in (
        (0, "exp1_remote_l2_write.png",
         "Remote L2 write (cross-die, traverses NV-HBI)"),
        (1, "exp1_local_l2_write.png",
         "Local L2 write (own die, no fabric hop)"),
    ):
        sel = [r for r in rows if i(r, "own_die") == own_die]
        if not sel:
            continue

        # group[(block_size, writer_partition)][sm] -> median GB/s over repeats
        group = defaultdict(lambda: defaultdict(list))
        for r in sel:
            group[(i(r, "block_size"), i(r, "writer_partition"))][
                i(r, "num_active_sm")].append(f(r, ycol))

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        block_sizes = sorted({k[0] for k in group})
        partitions = sorted({k[1] for k in group})
        cmap = plt.get_cmap("tab10")
        # colour = block_size (the series the question asks for)
        # dash    = writer partition, so both dies stay on one figure
        styles = {p: st for p, st in zip(partitions, ["-", "--", ":", "-."])}

        per_sm_ref = None
        for ci, bs in enumerate(block_sizes):
            for p in partitions:
                series = group.get((bs, p))
                if not series:
                    continue
                xs = sorted(series)
                ys = [median(series[x]) for x in xs]
                # Spell out that the dash split is the WRITING die, not the
                # target: reading it as own-vs-cross inverts the whole figure.
                lbl = f"block={bs}" + (f", writers on die{p}" if len(partitions) > 1 else "")
                ax.plot(xs, ys, marker="o", markersize=4, linestyle=styles[p],
                        color=cmap(ci % 10), label=lbl)
                if per_sm_ref is None and xs and xs[0] > 0:
                    per_sm_ref = ys[0] / xs[0]

        # Perfect scaling from the 1-SM rate. On log-log this is a straight line,
        # so where the curves peel off IS the saturation knee. On a linear y axis
        # it is invisible -- everything just looks like growth.
        if per_sm_ref:
            allx = sorted({x for g in group.values() for x in g})
            ax.plot(allx, [per_sm_ref * x for x in allx], color="0.45",
                    linestyle=":", linewidth=1.4, zorder=0,
                    label=f"ideal linear ({per_sm_ref:.0f} GB/s per SM)")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("number of writing SMs")
        ax.set_ylabel("write bandwidth (GB/s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


def plot_exp23(rows, stem, outdir):
    """Two panels: background BW and NVLink BW vs NVLink load.

    x is peer_blocks, the knob that sets how hard GPU1 injects. peer_blocks=0 is
    the clean baseline on the background panel; bg_sms=0 is the baseline on the
    NVLink panel. Flat lines here are the result, not a failure: neither P2P
    kernel stores nor copy-engine DMA moved the background by more than 0.1%.
    """
    if not rows:
        print(f"{stem}: no rows, skipping")
        return

    exp = i(rows[0], "exp")
    bg_local = i(rows[0], "bg_local")
    bg_mode = ("own-die background (control)" if bg_local
               else "cross-die background (bisection probe)")
    peer_desc = "peer -> FAR die" if exp == 2 else "peer -> NEAR die"

    bg = defaultdict(lambda: defaultdict(list))    # [bg_sms][peer_blocks]
    pr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, pb = i(r, "bg_sms"), i(r, "peer_blocks")
        if s > 0:
            bg[s][pb].append(f(r, "bg_GBps"))
        if pb > 0:
            pr[s][pb].append(f(r, "peer_GBps"))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    cmap = plt.get_cmap("viridis")
    sms_bg, sms_pr = sorted(bg), sorted(pr)

    def colour(s, allsms):
        if len(allsms) < 2:
            return cmap(0.5)
        return cmap(allsms.index(s) / (len(allsms) - 1))

    for s in sms_bg:
        xs = sorted(bg[s])
        ys = [median(bg[s][x]) for x in xs]
        axl.plot([max(x, 1) for x in xs], ys, marker="o", markersize=4,
                 color=colour(s, sms_bg), label=f"bg {s} SMs")
    axl.set_ylabel("background write bandwidth (GB/s)")
    axl.set_title("Background BW vs NVLink load")

    for s in sms_pr:
        xs = sorted(pr[s])
        ys = [median(pr[s][x]) for x in xs]
        axr.plot(xs, ys, marker="s", markersize=4, color=colour(s, sms_pr),
                 label=("no bg" if s == 0 else f"bg {s} SMs"))
    axr.set_ylabel("NVLink (peer) write bandwidth (GB/s)")
    axr.set_title("NVLink BW vs NVLink load")

    for ax in (axl, axr):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("NVLink load  (peer grid blocks; 1 = none)")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle(f"exp{exp}: {peer_desc},  {bg_mode}")
    fig.tight_layout()
    path = os.path.join(outdir, stem + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def summarise_exp23(rows, stem):
    """Print the two contrasts the figures are meant to show."""
    if not rows:
        return
    bg = defaultdict(lambda: defaultdict(list))
    pr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, pb = i(r, "bg_sms"), i(r, "peer_blocks")
        if s > 0:
            bg[s][pb].append(f(r, "bg_GBps"))
        if pb > 0:
            pr[s][pb].append(f(r, "peer_GBps"))
    if not bg:
        return
    smax = max(bg)
    pbs = sorted(p for p in bg[smax] if p > 0)
    if not pbs:
        return
    b0, b1 = median(bg[smax][0]), median(bg[smax][pbs[-1]])
    print(f"  {stem}: at {smax} bg SMs, NVLink 0 -> {pbs[-1]} blocks: "
          f"bg {b0:.1f} -> {b1:.1f} GB/s ({(b1/b0-1)*100:+.2f}%)")
    if 0 in pr and smax in pr:
        p0, p1 = median(pr[0][pbs[-1]]), median(pr[smax][pbs[-1]])
        print(f"  {stem}: peer alone {p0:.1f} -> with bg {p1:.1f} GB/s "
              f"({(p1/p0-1)*100:+.2f}%)")


def plot_exp4(rows, outdir):
    """x = message size, y = NCCL busbw, one line per background SM count.

    Solid = background crosses the dies, dashed = same background confined to
    its own die. The gap between them is the bisection effect; the drop both
    share against bg_sms=0 is NCCL simply losing SMs.
    """
    if not rows:
        print("exp4: no rows, skipping")
        return
    g = defaultdict(lambda: defaultdict(list))          # [(bg_local,bg_sms)][bytes]
    bgw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (i(r, "bg_local"), i(r, "bg_sms"))
        g[k][i(r, "msg_bytes")].append(f(r, "busbw_GBps"))
        if i(r, "bg_sms") > 0:
            bgw[k][i(r, "msg_bytes")].append(f(r, "bg_GBps"))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    sms = sorted({k[1] for k in g})
    cmap = plt.get_cmap("viridis")
    for (loc, s), series in sorted(g.items()):
        xs = sorted(series)
        ys = [median(series[x]) for x in xs]
        col = cmap(sms.index(s) / max(len(sms) - 1, 1))
        axl.plot(xs, ys, marker="o", markersize=4, color=col,
                 linestyle="--" if loc else "-",
                 label=f"bg {s} SMs" + (" (own-die)" if loc else ""))
    axl.set_ylabel("NCCL busbw (GB/s)")
    axl.set_title("Collective throughput vs message size")
    axl.set_xscale("log", base=2)

    for (loc, s), series in sorted(bgw.items()):
        xs = sorted(series)
        ys = [median(series[x]) for x in xs]
        col = cmap(sms.index(s) / max(len(sms) - 1, 1))
        axr.plot(xs, ys, marker="s", markersize=4, color=col,
                 linestyle="--" if loc else "-",
                 label=f"bg {s} SMs" + (" (own-die)" if loc else ""))
    axr.set_ylabel("background write bandwidth (GB/s)")
    axr.set_title("What the background gave up")
    axr.set_xscale("log", base=2)

    for ax in (axl, axr):
        ax.set_xlabel("per-pair message size (bytes)")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle("exp4: NCCL all-to-all under NV-HBI contention "
                 "(buffers are die-agnostic; NCCL also competes for SMs)")
    fig.tight_layout()
    path = os.path.join(outdir, "exp4_nccl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_dual(rows, outdir):
    """x = reader SMs, y = write / read / their sum, at fixed writer count.

    Both payloads travel the same direction, so the sum is what that direction
    carried. The dashed line is writes with no readers at all: if the sum never
    reaches it, a second source does not add bandwidth.
    """
    if not rows:
        print("dualdir: no rows, skipping")
        return
    loc = [r for r in rows if i(r, "r_local") == 0]
    if not loc:
        return
    w_max = max(i(r, "w_sms") for r in loc)
    sel = [r for r in loc if i(r, "w_sms") == w_max]
    if not sel:
        return

    series = defaultdict(lambda: defaultdict(list))
    for r in sel:
        rs = i(r, "r_sms")
        series["write"][rs].append(f(r, "write_GBps"))
        series["read"][rs].append(f(r, "read_GBps"))
        series["total"][rs].append(f(r, "total_GBps"))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for name, col in (("write", "tab:blue"), ("read", "tab:orange"),
                      ("total", "tab:green")):
        xs = sorted(series[name])
        ys = [median(series[name][x]) for x in xs]
        ax.plot([max(x, 1) for x in xs], ys, marker="o", markersize=5,
                color=col, label=name)
    solo = series["write"].get(0)
    if solo:
        ax.axhline(median(solo), color="tab:blue", linestyle="--", linewidth=1.2,
                   label=f"writes alone ({median(solo):.0f} GB/s)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("reader SMs on the far die  (1 = none)")
    ax.set_ylabel("bandwidth on the A->B direction (GB/s)")
    ax.set_title(f"Two payload sources on one direction ({w_max} writer SMs)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "dualdir.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("-o", "--outdir", default=None,
                    help="where to write PNGs (default: the session directory)")
    args = ap.parse_args()

    if not os.path.isdir(args.session_dir):
        sys.exit(f"not a directory: {args.session_dir}")
    outdir = args.outdir or args.session_dir
    os.makedirs(outdir, exist_ok=True)

    print(f"session: {args.session_dir}")
    exp1 = load_rows(args.session_dir, "exp1", EXP1_HEADER_HINT)
    print(f"exp1: {len(exp1)} rows")
    plot_exp1(exp1, outdir)

    for stem in ("exp2_bglocal0", "exp2_bglocal1",
                 "exp3_bglocal0", "exp3_bglocal1"):
        rows = load_rows(args.session_dir, stem, EXP23_HEADER)
        print(f"{stem}: {len(rows)} rows")
        plot_exp23(rows, stem, outdir)
        summarise_exp23(rows, stem)

    for stem in ("exp4_bglocal0", "exp4_bglocal1", "exp4_nccl"):
        rows = load_rows(args.session_dir, stem, EXP4_HEADER)
        if rows:
            print(f"{stem}: {len(rows)} rows")
    exp4 = []
    for stem in ("exp4_bglocal0", "exp4_bglocal1", "exp4_nccl"):
        exp4 += load_rows(args.session_dir, stem, EXP4_HEADER)
    plot_exp4(exp4, outdir)

    dual = load_rows(args.session_dir, "dualdir", DUAL_HEADER)
    print(f"dualdir: {len(dual)} rows")
    plot_dual(dual, outdir)


if __name__ == "__main__":
    main()
