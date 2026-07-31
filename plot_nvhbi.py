#!/usr/bin/env python3
"""plot_nvhbi.py -- figures for one NV-HBI session directory.

Reads a session produced by run_session.sh (or the raw .log files) and writes:

  exp1_remote_l2_write.png   x = #SMs (linear, ticked every 10), y = TB/s
  exp1_local_l2_write.png    same, for own-die writes
  exp2_bglocal0.png          x = NVLink load in TB/s, y = background TB/s,
  exp2_bglocal1.png          one line per background SM count
  exp3_bglocal0.png
  exp3_bglocal1.png
  dualdir.png                x = reader SMs, y = TB/s on the A->B direction
  peerl2.png                 peer read latency (warm vs cold) and throughput

Axes are linear throughout and throughput is reported in TB/s.

Prefers <name>.csv; falls back to extracting the "CFG," rows from <name>.log,
so it works on a directory that only has logs.

Usage:
    python3 plot_nvhbi.py SESSION_DIR [-o OUTDIR]
"""

import argparse
import csv
import io
import os
import re
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
                "peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl")
PL2_HEADER = ("kind,die,is_far,state,cv,corun,rep,peer_cyc,local_cyc,"
              "peer_GBps,co_GBps,chunks")
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


# The peer kernel's block size is fixed in nvhbi_exp23_peer.cu and is not in the
# CSV, so the axis label has to carry it from here.
PEER_BLOCK_THREADS = 128


def peer_gpu_sms(session_dir, stem, default=148):
    """SM count of the injecting GPU, for the exp2/3 grid-size axis label.

    Not a CSV column, so it comes from the run log's probe line. Both GPUs in
    these sessions are the same part, and the probe only prints gpu0.
    """
    log_path = os.path.join(session_dir, stem + ".log")
    if os.path.exists(log_path):
        for line in open(log_path):
            m = re.search(r"SMs=(\d+)", line)
            if m:
                return int(m.group(1))
    return default


def sm_ticks(ax, allx):
    """Linear SM axis, ticked every 10."""
    top = max(allx) if allx else 0
    ax.set_xlim(0, top + 5)
    ax.set_xticks(list(range(0, int(top) + 11, 10)))


# exp1 sweeps two writing dies and (optionally) several blocks-per-SM values.
# The figure fixes both so that block_size is the only series left: mixing the
# writing die into the same axes doubled every curve and made the block_size
# comparison unreadable.
EXP1_WRITER_PARTITION = 1
EXP1_BLOCKS_PER_SM = 32


def plot_exp1(rows, outdir):
    """One figure per die target: x = #SMs, y = throughput in TB/s.

    One line per block_size, at a single writing die and a single blocks-per-SM.
    """
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
        sel = [r for r in rows
               if i(r, "own_die") == own_die
               and i(r, "writer_partition") == EXP1_WRITER_PARTITION
               and i(r, "num_blocks_per_sm") == EXP1_BLOCKS_PER_SM]
        if not sel:
            print(f"exp1: no rows for own_die={own_die}, "
                  f"writers on die{EXP1_WRITER_PARTITION}, "
                  f"{EXP1_BLOCKS_PER_SM} blocks/SM -- skipping")
            continue

        group = defaultdict(lambda: defaultdict(list))   # [block_size][sm] -> GB/s
        for r in sel:
            group[i(r, "block_size")][i(r, "num_active_sm")].append(f(r, ycol))

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        cmap = plt.get_cmap("tab10")
        for ci, bs in enumerate(sorted(group)):
            series = group[bs]
            xs = sorted(series)
            ys = [median(series[x]) / 1000.0 for x in xs]     # GB/s -> TB/s
            ax.plot(xs, ys, marker="o", markersize=4, color=cmap(ci % 10),
                    label=f"block size = {bs}")

        allx = sorted({x for g in group.values() for x in g})
        sm_ticks(ax, allx)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of SMs")
        ax.set_ylabel("Throughput (TB/s)")
        ax.set_title(f"{title}\nwriters on die{EXP1_WRITER_PARTITION}, "
                     f"{EXP1_BLOCKS_PER_SM} blocks/SM", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


def plot_exp23(rows, stem, outdir, peer_sms=148):
    """Two panels, and the x axes are deliberately different.

    LEFT is the experiment: x is the NVLink load in GB/s, y is the background's
    remaining bandwidth. The load is the peer bandwidth measured at that grid
    size WITH NO BACKGROUND -- an offered load, so it stays an independent
    variable. The first version of this figure put peer_blocks on x, which is a
    grid size, not a load, and hid the real problem: every grid size from 16
    blocks up delivered 634-645 GB/s, so the whole axis sat past saturation and
    four points were plotted where there was really only one.

    RIGHT is the diagnostic for that: x is the number of injecting warps, y is
    what they achieved. Where that curve goes flat is where the left panel stops
    carrying information, so read it first.
    """
    if not rows:
        print(f"{stem}: no rows, skipping")
        return

    exp = i(rows[0], "exp")
    bg_local = i(rows[0], "bg_local")
    ovl = any(i(r, "peer_ovl") == 1 for r in rows)
    bg_mode = ("own-die background (control)" if bg_local
               else "cross-die background")
    peer_desc = "peer -> FAR die" if exp == 2 else "peer -> NEAR die"

    bg = defaultdict(lambda: defaultdict(list))    # [bg_sms][peer_blocks]
    pr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, pb = i(r, "bg_sms"), i(r, "peer_blocks")
        if s > 0:
            bg[s][pb].append(f(r, "bg_GBps"))
        if pb > 0:
            pr[s][pb].append(f(r, "peer_GBps"))

    # peer_blocks -> offered NVLink load in GB/s (measured with no background).
    # Falls back to the smallest background if the bg_sms=0 row is missing.
    ref = pr.get(0) or (pr[min(pr)] if pr else {})
    offered = {pb: median(v) for pb, v in ref.items()}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    cmap = plt.get_cmap("viridis")
    sms_bg, sms_pr = sorted(bg), sorted(pr)

    def colour(s, allsms):
        if len(allsms) < 2:
            return cmap(0.5)
        return cmap(allsms.index(s) / (len(allsms) - 1))

    for s in sms_bg:
        pts = sorted((offered.get(pb, 0.0) / 1000.0, median(bg[s][pb]) / 1000.0)
                     for pb in bg[s])
        if not pts:
            continue
        axl.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=4,
                 color=colour(s, sms_bg), label=f"bg {s} SMs")
    axl.set_xlabel("NVLink load offered by GPU1 (TB/s, measured with no background)")
    axl.set_ylabel("Throughput (TB/s)")
    axl.set_title("Does NVLink traffic cost the background anything?")
    axl.set_xlim(left=0)

    # x = the peer grid size itself, i.e. how many blocks GPU1 injects with.
    for s in sms_pr:
        xs = sorted(pr[s])
        ys = [median(pr[s][x]) / 1000.0 for x in xs]
        axr.plot(xs, ys, marker="s", markersize=4,
                 color=colour(s, sms_pr),
                 label=("no bg" if s == 0 else f"bg {s} SMs"))
    axr.set_xlabel(f"Number of blocks ({peer_sms} SM, "
                   f"block size {PEER_BLOCK_THREADS})")
    axr.set_ylabel("Throughput (TB/s)")
    axr.set_title("Where the NVLink load axis saturates")
    axr.set_xlim(left=0)

    for ax in (axl, axr):
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle(f"exp{exp}: {peer_desc},  {bg_mode}"
                 + ("   [peer and background on the SAME chunks]" if ovl else ""))
    fig.tight_layout()
    path = os.path.join(outdir, stem + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_peerl2(rows, outdir):
    """Is a peer read served from GPU0's L2 or from GPU0's HBM?

    Left: latency, warm vs cold, peer against the GPU0-local reference measured
    with the identical kernel. The local pair sets the scale -- it is what an L2
    hit and an HBM miss cost on this allocation. If the peer bars move by a
    similar margin, GPU1 is being served by GPU0's L2; if the peer bars are flat
    across warm/cold, it is not.

    Right: bandwidth for the resident (fits in L2) and streaming (>> L2)
    footprints, with and without GPU0 co-running on the same chunks.
    """
    if not rows:
        print("peerl2: no rows, skipping")
        return

    lat = [r for r in rows if i(r, "kind") == 0]
    bw = [r for r in rows if i(r, "kind") == 1]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 5.0))

    if lat:
        # one group per (die, cv); bars are peer/local x cold/warm
        keys = sorted({(i(r, "die"), i(r, "cv")) for r in lat})
        far = {i(r, "die"): i(r, "is_far") for r in lat}
        width = 0.2
        for gi, k in enumerate(keys):
            die, cv = k
            for bi_, (col, state, colour_) in enumerate((
                    ("peer_cyc", 0, "tab:red"), ("peer_cyc", 1, "tab:orange"),
                    ("local_cyc", 0, "tab:blue"), ("local_cyc", 1, "tab:cyan"))):
                vals = [f(r, col) for r in lat
                        if i(r, "die") == die and i(r, "cv") == cv
                        and i(r, "state") == state]
                if not vals:
                    continue
                axl.bar(gi + (bi_ - 1.5) * width, median(vals), width,
                        color=colour_,
                        label=(f"{'GPU1 peer' if col.startswith('peer') else 'GPU0 local'}"
                               f", {'warm L2' if state else 'cold (HBM)'}")
                        if gi == 0 else None)
        axl.set_xticks(range(len(keys)))
        axl.set_xticklabels([f"die{d}{' FAR' if far.get(d) else ' NEAR'}\n"
                             f"{'ld.cv' if c else 'ld.cg'}" for d, c in keys],
                            fontsize=8)
        axl.set_ylabel("per-access latency (SM cycles)")
        axl.set_title("Peer read latency: warm L2 vs cold")
        axl.legend(fontsize=7)
        axl.grid(True, alpha=0.3, axis="y")

    if bw:
        labels, vals = [], []
        for r_key in sorted({(i(r, "die"), i(r, "state"), i(r, "cv"), i(r, "corun"))
                             for r in bw}):
            die, state, cv, corun = r_key
            sel = [f(r, "peer_GBps") / 1000.0 for r in bw
                   if (i(r, "die"), i(r, "state"), i(r, "cv"), i(r, "corun")) == r_key]
            if not sel:
                continue
            labels.append(f"die{die} {'res' if state else 'stream'} "
                          f"{'cv' if cv else 'cg'} co{corun}")
            vals.append(median(sel))
        axr.barh(range(len(vals)), vals, color="tab:green")
        axr.set_yticks(range(len(labels)))
        axr.set_yticklabels(labels, fontsize=7)
        axr.set_xlabel("GPU1 peer read Throughput (TB/s)")
        axr.set_title("Peer read bandwidth  (res = fits L2, stream = >> L2)")
        axr.grid(True, alpha=0.3, axis="x")

    fig.suptitle("peerl2: where GPU1's reads of GPU0 are served from")
    fig.tight_layout()
    path = os.path.join(outdir, "peerl2.png")
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
        ys = [median(series[x]) / 1000.0 for x in xs]
        col = cmap(sms.index(s) / max(len(sms) - 1, 1))
        axl.plot(xs, ys, marker="o", markersize=4, color=col,
                 linestyle="--" if loc else "-",
                 label=f"bg {s} SMs" + (" (own-die)" if loc else ""))
    axl.set_ylabel("NCCL busbw -- Throughput (TB/s)")
    axl.set_title("Collective throughput vs message size")
    axl.set_xscale("log", base=2)

    for (loc, s), series in sorted(bgw.items()):
        xs = sorted(series)
        ys = [median(series[x]) / 1000.0 for x in xs]
        col = cmap(sms.index(s) / max(len(sms) - 1, 1))
        axr.plot(xs, ys, marker="s", markersize=4, color=col,
                 linestyle="--" if loc else "-",
                 label=f"bg {s} SMs" + (" (own-die)" if loc else ""))
    axr.set_ylabel("Throughput (TB/s)")
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
    allx = sorted(series["total"])
    for name, col in (("write", "tab:blue"), ("read", "tab:orange"),
                      ("total", "tab:green")):
        xs = sorted(series[name])
        ys = [median(series[name][x]) / 1000.0 for x in xs]
        ax.plot(xs, ys, marker="o", markersize=5, color=col, label=name)
    solo = series["write"].get(0)
    if solo:
        ax.axhline(median(solo) / 1000.0, color="tab:blue", linestyle="--",
                   linewidth=1.2, label=f"writes alone ({median(solo) / 1000.0:.2f} TB/s)")
    sm_ticks(ax, allx)
    ax.set_xlabel("reader SMs on the far die")
    ax.set_ylabel("Throughput on the A->B direction (TB/s)")
    ax.set_title(f"Two payload sources on one direction ({w_max} writer SMs)")
    ax.grid(True, alpha=0.3)
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
        plot_exp23(rows, stem, outdir,
                   peer_sms=peer_gpu_sms(args.session_dir, stem))
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

    pl2 = load_rows(args.session_dir, "peerl2", PL2_HEADER)
    print(f"peerl2: {len(pl2)} rows")
    plot_peerl2(pl2, outdir)


if __name__ == "__main__":
    main()
