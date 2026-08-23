#!/usr/bin/env python3
"""plot_nvhbi.py -- figures for one NV-HBI session directory.

Reads a session produced by run_session.sh and writes:

  exp1_remote_l2_write.png   cross-die write bandwidth vs #SMs
  exp1_local_l2_write.png    own-die write bandwidth vs #SMs (the control)
  dualdir.png                one direction, writes + crossing reads
  dualdir_local.png          same with the reads kept local (the control)
  exp2_bglocal0.png          peer -> FAR die, background crossing
  exp2_bglocal1.png          peer -> FAR die, background own-die
  exp3_bglocal0.png          peer -> NEAR die (control), background crossing
  exp3_bglocal1.png          peer -> NEAR die, background own-die

Axes are linear and throughput is in TB/s throughout.

Prefers <name>.csv, falls back to the "CFG," rows of <name>.log, so it also
works on a directory that only has logs.

Usage:
    python3 plot_nvhbi.py SESSION_DIR [-o OUTDIR]
"""

import argparse
import csv
import io
import os
import re
import sys
from collections import Counter, defaultdict
from statistics import median
import matplotlib.ticker as ticker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column sets are addressed by NAME everywhere; these are only fallbacks for a
# log that has no "# CFG," header line.
EXP1_HEADER_HINT = "writer_partition,own_die,num_active_sm"
EXP23_HEADER = ("exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,"
                "peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl,"
                "peer_bsize,bg_rd_GBps,bg_r_sms")
DUAL_HEADER = ("w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps,"
               "dieB_GBps")

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


def peer_grid_info(session_dir, stem, default_sms=148):
    """(SM count, peer blocks/SM) for the exp2/3 injection-axis label.

    Neither is a CSV column. The SM count comes from the probe line; the
    blocks-per-SM is the REQUESTED value from the config line, which is what the
    label should say -- the CSV's peer_blocks holds the grid actually launched,
    and that gets clamped at the top of the block-size sweep where 32 blocks/SM
    would need 4096 threads against a 2048 limit.
    """
    sms, bps = default_sms, None
    log_path = os.path.join(session_dir, stem + ".log")
    if os.path.exists(log_path):
        for line in open(log_path):
            if bps is None:
                m = re.search(r"peer grid: (\d+) blocks per SM", line)
                if m:
                    bps = int(m.group(1))
            m = re.search(r"SMs=(\d+)", line)
            if m:
                sms = int(m.group(1))
    return sms, bps


def sm_ticks(ax, allx):
    """Linear SM axis, ticked every 10."""
    top = max(allx) if allx else 0
    ax.set_xlim(0, top + 5)
    ax.set_xticks(list(range(0, int(top) + 11, 10)))


# exp1 sweeps two writing dies and (optionally) several blocks-per-SM values.
# The figure fixes both so that block_size is the only series left: mixing the
# writing die into the same axes doubled every curve and made the block_size
# comparison unreadable.
def plot_exp1(rows, outdir):
    """One figure per target die: x = #SMs, y = throughput in TB/s.

    exp1 sweeps both writer partitions and several block sizes, so fix one
    partition -- whichever has the most rows -- and draw one line per block
    size. sampled_GBps is the column to compare against nvhbi_dualdir and
    nvhbi_exp23_peer; slope_GBps is timed by the slowest warp instead.
    """
    if not rows:
        print("exp1: no rows, skipping")
        return

    ycol = ("sampled_GBps" if any(f(r, "sampled_GBps") > 0 for r in rows)
            else "slope_GBps")
    part = Counter(i(r, "writer_partition") for r in rows).most_common(1)[0][0]
    nbps = Counter(i(r, "num_blocks_per_sm") for r in rows).most_common(1)[0][0]

    for own_die, fname, title in (
        (0, "exp1_remote_l2_write.png",
         "Cross-die write (traverses NV-HBI)"),
        (1, "exp1_local_l2_write.png",
         "Own-die write (control: no fabric hop)"),
    ):
        sel = [r for r in rows
               if i(r, "own_die") == own_die
               and i(r, "writer_partition") == part
               and i(r, "num_blocks_per_sm") == nbps]
        if not sel:
            print(f"exp1: nothing for own_die={own_die} -- skipping")
            continue

        group = defaultdict(lambda: defaultdict(list))   # [block_size][sm]
        for r in sel:
            group[i(r, "block_size")][i(r, "num_active_sm")].append(f(r, ycol))

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        cmap = plt.get_cmap("tab10")
        for ci, bs in enumerate(sorted(group)):
            xs = sorted(group[bs])
            ys = [median(group[bs][x]) / 1000.0 for x in xs]
            ax.plot(xs, ys, marker="o", markersize=4, color=cmap(ci % 10),
                    label=f"block size = {bs}")

        sm_ticks(ax, sorted({x for g in group.values() for x in g}))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of writing SMs")
        ax.set_ylabel("Throughput (TB/s)")
        ax.set_title(f"{title}\nwriters on die{part}, {nbps} blocks/SM "
                     f"({ycol})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


def plot_exp23(rows, stem, outdir, peer_sms=148, peer_bps=None):
    """Two panels, with deliberately different x axes.

    LEFT is the experiment: x is the NVLink load offered by GPU1, y is what the
    background still gets. The load on x is the peer bandwidth measured at that
    block size WITH NO BACKGROUND, so it stays an independent variable. One line
    per background configuration; a background with readers is plotted as
    write + read, because both payloads are on the fabric.

    RIGHT says where that x axis stops carrying information: x is the peer block
    size, y is what the peer achieved. Where this goes flat, the left panel's
    rightmost points are all the same offered load. Read it first.
    """
    if not rows:
        print(f"{stem}: no rows, skipping")
        return

    exp = i(rows[0], "exp")
    bg_local = i(rows[0], "bg_local")
    ovl = any(i(r, "peer_ovl") == 1 for r in rows)
    bg_mode = ("own-die background (control)" if bg_local
               else "cross-die background")
    peer_desc = "peer -> FAR die" if exp == 2 else "peer -> NEAR die (control)"

    def bg_key(r):
        return (i(r, "bg_sms"), i(r, "bg_r_sms"))

    def bg_label(k):
        w, rd = k
        return f"bg {w}w" + (f"+{rd}r" if rd else "")

    # background throughput on the fabric = its writes plus its crossing reads
    def bg_val(r):
        return f(r, "bg_GBps") + f(r, "bg_rd_GBps")

    bg = defaultdict(lambda: defaultdict(list))   # [bg config][block size]
    pr = defaultdict(lambda: defaultdict(list))   # [bg config][block size]
    for r in rows:
        k, bs = bg_key(r), i(r, "peer_bsize")
        if k != (0, 0):
            bg[k][bs].append(bg_val(r))
        if bs > 0:
            pr[k][bs].append(f(r, "peer_GBps"))

    # block size -> offered NVLink load, measured with no background at all
    ref = pr.get((0, 0)) or (pr[sorted(pr)[0]] if pr else {})
    offered = {bs: median(v) for bs, v in ref.items()}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    cmap = plt.get_cmap("viridis")
    keys_bg, keys_pr = sorted(bg), sorted(pr)

    def colour(k, keys):
        if len(keys) < 2:
            return cmap(0.5)
        return cmap(keys.index(k) / (len(keys) - 1))

    for k in keys_bg:
        pts = sorted((offered.get(bs, 0.0) / 1000.0, median(bg[k][bs]) / 1000.0)
                     for bs in bg[k])
        if not pts:
            continue
        axl.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                 markersize=4, color=colour(k, keys_bg), label=bg_label(k))
    axl.set_xlabel("NVLink load offered by GPU1 (TB/s, measured with no background)")
    axl.set_ylabel("Background throughput (TB/s)")
    axl.set_title("Does NVLink traffic cost the background anything?")
    axl.set_xlim(left=0)

    # Power-of-two ladder, so log2 x with plain tick labels.
    for k in keys_pr:
        xs = sorted(pr[k])
        ys = [median(pr[k][x]) / 1000.0 for x in xs]
        axr.plot(xs, ys, marker="s", markersize=4, color=colour(k, keys_pr),
                 label=("no bg" if k == (0, 0) else bg_label(k)))
    axr.set_xscale("log", base=2)
    axr.xaxis.set_major_formatter(ticker.ScalarFormatter())
    axr.ticklabel_format(style="plain", axis="x")
    grid = f"{peer_bps} blocks/SM" if peer_bps else "clamped grid"
    axr.set_xlabel(f"Peer block size ({peer_sms} SMs, {grid})")
    axr.set_ylabel("Peer throughput (TB/s)")
    axr.set_title("Where the NVLink load axis saturates")

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


def summarise_exp23(rows, stem):
    """Print, per background configuration, what the peer cost it."""
    if not rows:
        return
    bg = defaultdict(lambda: defaultdict(list))
    pr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k, bs = (i(r, "bg_sms"), i(r, "bg_r_sms")), i(r, "peer_bsize")
        if k != (0, 0):
            bg[k][bs].append(f(r, "bg_GBps") + f(r, "bg_rd_GBps"))
        if bs > 0:
            pr[k][bs].append(f(r, "peer_GBps"))
    if not bg:
        return
    bs_max = max(b for cfg in bg.values() for b in cfg if b > 0)
    for k in sorted(bg):
        if 0 not in bg[k] or bs_max not in bg[k]:
            continue
        b0, b1 = median(bg[k][0]), median(bg[k][bs_max])
        peer = median(pr[k][bs_max]) if bs_max in pr.get(k, {}) else 0.0
        print(f"  {stem}: bg {k[0]}w+{k[1]}r  {b0:7.1f} -> {b1:7.1f} GB/s "
              f"({(b1 / b0 - 1) * 100:+5.2f}%)   peer {peer:6.1f}   "
              f"crossing {b1 + peer:7.1f}")


def plot_dual(rows, outdir, stem="dualdir", title="crossing reads"):
    """x = reader SMs, y = write / read / their sum, at the largest writer count.

    Both payloads travel A->B, so their sum is what that direction carried --
    but only when the reads actually cross. With r_local=1 nothing crosses on
    the read side, and the figure then shows the write curve alone against the
    read load it is being asked to tolerate. The dashed line is writes with no
    readers: whether the write curve stays on it is the whole point.
    """
    if not rows:
        print(f"{stem}: no rows, skipping")
        return
    r_local = i(rows[0], "r_local")
    w_max = max(i(r, "w_sms") for r in rows)
    sel = [r for r in rows if i(r, "w_sms") == w_max]
    if not sel:
        return

    series = defaultdict(lambda: defaultdict(list))
    for r in sel:
        rs = i(r, "r_sms")
        series["write"][rs].append(f(r, "write_GBps"))
        series["read"][rs].append(f(r, "read_GBps"))
        if not r_local:
            series["write + read"][rs].append(f(r, "total_GBps"))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for name, col in (("write", "tab:blue"), ("read", "tab:orange"),
                      ("write + read", "tab:green")):
        if name not in series:
            continue
        xs = sorted(series[name])
        ys = [median(series[name][x]) / 1000.0 for x in xs]
        ax.plot(xs, ys, marker="o", markersize=5, color=col, label=name)

    solo = series["write"].get(0)
    if solo:
        ax.axhline(median(solo) / 1000.0, color="tab:blue", linestyle="--",
                   linewidth=1.2,
                   label=f"writes alone ({median(solo) / 1000.0:.2f} TB/s)")

    sm_ticks(ax, sorted(series["write"]))
    ax.set_xlabel("reader SMs")
    ax.set_ylabel("Throughput (TB/s)")
    ax.set_title(f"One direction, two payload sources -- {title}\n"
                 f"{w_max} writer SMs, writes cross the die boundary")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, stem + ".png")
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
        _sms, _bps = peer_grid_info(args.session_dir, stem)
        plot_exp23(rows, stem, outdir, peer_sms=_sms, peer_bps=_bps)
        summarise_exp23(rows, stem)

    for stem, title in (("dualdir", "crossing reads"),
                        ("dualdir_local", "local reads (control)")):
        rows = load_rows(args.session_dir, stem, DUAL_HEADER)
        print(f"{stem}: {len(rows)} rows")
        plot_dual(rows, outdir, stem, title)


if __name__ == "__main__":
    main()
