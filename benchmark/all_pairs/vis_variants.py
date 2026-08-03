"""Alternative visualizations of the all-pairs benchmark average data.

Generates several "average across machines" plots using different layouts so the
user can compare which one is clearest.  Each variant is saved into plots/ with a
descriptive name.

Variants
--------
Variant A : naive / binned split into two side-by-side panels
Variant B : facet grid  -- one panel per benchmark, machines as the lines
Variant C : speedup vs ``NaiveFor`` baseline on a shared axis
Variant D : heatmap (benchmark x agents) of log10 time, one per config family
Variant E : grouped bar chart (benchmark x agents), one panel per family
Variant F : family bands -- shade min/max of each family on ONE plot to show the \
            naive <-> binned crossover with no per-variant line overlap
Variant G : naive vs binned scatter -- per-pair speedup/crossover, zero overlap
Variant H : best-of-family lines -- cleanest possible naive <-> binned crossover
Variant I : grouped by precision family (For / Parallel / Spread) -- less overlap
Variant J : speedup-ratio lines (binned/naive) vs agents -- crossover at ratio 1
Variant K : box plot of each family per agent count (distribution, no lines)
Variant L : all 13 lines subdued + best naive / best binned emphasised
Variant M : parallel slope chart -- two vertical axes (naive | binned), one wrap per variant
Variant N : 100% stacked area -- family share of runtime vs agents (composition crossover)
Variant O : divergence filled area -- log10(binned/naive) as shaded area above/below zero
Variant P : single speedup-ratio heatmap -- variant x agents, colour = binned/naive ratio
Variant S : diverging waterfall bars -- best binned vs best naive speedup per agents
Variant T : range/bullet chart -- per agent: spread band for Naive & Binned, variants marked
Variant U : radar chart -- Naive vs Binned across access patterns at a chosen size
Variant V : stacked area chart -- all variants stacked (runtime vs agents)
Variant W : stream graph -- symmetric stacked area of all variants
Variant X : gantt/range bars -- one horizontal range bar per variant spanning its times
Variant Y : enhanced diverging -- BOTH families shown, non-best/worst variants marked
Variant Z : family bands + all benchmarks faint inside, best & worst named on plot
Variant AA: small-multiples -- all 13 benchmarks, one clean panel each, shared axes
Variant AB: ONE plot -- all 13 styled benchmark lines + family bands underneath\n            for the speedup envelope
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# styling shared with vis.py
# ---------------------------------------------------------------------------
marker_for_benchmark = {
    "NaiveFor": "o", "NaiveForDouble": "v", "NaiveForSymmetric": "<",
    "NaiveParallelReduce": "s", "NaiveParallelReduceDouble": "P",
    "NaiveParallelReduceSymmetric": "X", "NaiveSpread": "^",
    "BinnedReduceFor": "D", "BinnedReduceForDouble": "d",
    "BinnedReduceForSymmetric": "p", "BinnedReduceParallel": "h",
    "BinnedReduceParallelDouble": "H", "BinnedReduceParallelSymmetric": "*",
}

style_for_benchmark = {
    "NaiveFor": {"linestyle": "-", "color": "tab:blue", "alpha": 0.5, "linewidth": 1.5},
    "NaiveForDouble": {"linestyle": "--", "color": "tab:blue", "alpha": 0.7, "linewidth": 2.0},
    "NaiveForSymmetric": {"linestyle": ":", "color": "tab:blue", "alpha": 1.0, "linewidth": 2.5},
    "NaiveParallelReduce": {"linestyle": "-", "color": "tab:orange", "alpha": 0.5, "linewidth": 1.5},
    "NaiveParallelReduceDouble": {"linestyle": "--", "color": "tab:orange", "alpha": 0.7, "linewidth": 2.0},
    "NaiveParallelReduceSymmetric": {"linestyle": ":", "color": "tab:orange", "alpha": 1.0, "linewidth": 2.5},
    "NaiveSpread": {"linestyle": "-", "color": "tab:green", "alpha": 0.5, "linewidth": 1.5},
    "BinnedReduceFor": {"linestyle": "-", "color": "tab:purple", "alpha": 0.5, "linewidth": 1.5},
    "BinnedReduceForDouble": {"linestyle": "--", "color": "tab:purple", "alpha": 0.7, "linewidth": 2.0},
    "BinnedReduceForSymmetric": {"linestyle": ":", "color": "tab:purple", "alpha": 1.0, "linewidth": 2.5},
    "BinnedReduceParallel": {"linestyle": "-", "color": "tab:brown", "alpha": 0.5, "linewidth": 1.5},
    "BinnedReduceParallelDouble": {"linestyle": "--", "color": "tab:brown", "alpha": 0.7, "linewidth": 2.0},
    "BinnedReduceParallelSymmetric": {"linestyle": ":", "color": "tab:brown", "alpha": 1.0, "linewidth": 2.5},
}

# Palette used when each line is a *machine* instead of a benchmark.
MACHINE_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cleaned = []
    for r in rows:
        if not r.get("machine") or not r.get("agents"):
            continue
        try:
            agents_int = int(r["agents"])
            time_val = float(r.get("time_per_step_ms") or 0.0)
        except (ValueError, TypeError):
            continue
        r["_agents_int"] = agents_int
        r["_time"] = time_val
        cleaned.append(r)
    return cleaned


def average_over_machines(rows):
    """Return {benchmark: {agents: mean_time}} averaged over all machines."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["benchmark"], r["_agents_int"])].append(r["_time"])
    out = defaultdict(dict)
    for (bench, agents), vals in acc.items():
        out[bench][agents] = mean(vals)
    return dict(out)


def sorted_agents(bench):
    return sorted(bench.keys())


# ---------------------------------------------------------------------------
# Variant A -- naive / binned split into two panels
# ---------------------------------------------------------------------------
def plot_variant_a(avg, out_dir):
    naive = {k: v for k, v in avg.items() if k.startswith("Naive")}
    binned = {k: v for k, v in avg.items() if k.startswith("Binned")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def draw(ax, data, title):
        for bench, bench_data in sorted(data.items()):
            agents = sorted_agents(bench_data)
            times = [bench_data[a] for a in agents]
            sty = style_for_benchmark.get(bench, {})
            ax.plot(agents, times, marker=marker_for_benchmark.get(bench, "o"),
                    markersize=6, linewidth=sty.get("linewidth", 2),
                    linestyle=sty.get("linestyle", "--"),
                    color=sty.get("color", None), alpha=sty.get("alpha", 0.8),
                    label=bench)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("agents")
        ax.set_ylabel("time_per_step_ms")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    draw(ax1, naive, "Average — Naive")
    draw(ax2, binned, "Average — Binned")
    fig.tight_layout()
    p = out_dir / "variantA_split_naive_binned.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant B -- facet grid: one panel per benchmark, machines as lines
# ---------------------------------------------------------------------------
def plot_variant_b(rows, out_dir):
    benchmarks = sorted({r["benchmark"] for r in rows})
    machines = sorted({r["machine"] for r in rows})

    ncols = 4
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, bench in zip(axes, benchmarks):
        for i, machine in enumerate(machines):
            pts = [(r["_agents_int"], r["_time"]) for r in rows
                   if r["benchmark"] == bench and r["machine"] == machine]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", markersize=4, linewidth=1.2,
                    color=MACHINE_COLORS[i % len(MACHINE_COLORS)], label=machine)
        ax.set_title(bench, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # hide any unused panels
    for ax in axes[len(benchmarks):]:
        ax.set_visible(False)

    # one shared legend for all machines
    handles = []
    for i, machine in enumerate(machines):
        handles.append(Line2D([], [], color=MACHINE_COLORS[i % len(MACHINE_COLORS)],
                              marker="o", linestyle="-", label=machine))
    fig.legend(handles=handles, loc="lower center", ncol=len(machines), fontsize=8,
               frameon=False)
    fig.suptitle("Per-benchmark scaling across machines", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    p = out_dir / "variantB_facet_per_benchmark.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant C -- speedup relative to NaiveFor baseline (average data)
# ---------------------------------------------------------------------------
def plot_variant_c(avg, out_dir):
    base_name = "NaiveFor"
    if base_name not in avg:
        print(f"Variant C: baseline {base_name!r} missing, skipping")
        return
    base_agents = avg[base_name]

    fig, ax = plt.subplots(figsize=(11, 7))
    for bench, bench_data in sorted(avg.items()):
        if bench == base_name:
            continue
        agents = sorted_agents(bench_data)
        # only compare where the baseline has data too
        agents = [a for a in agents if a in base_agents]
        ratio = [bench_data[a] / base_agents[a] for a in agents]
        sty = style_for_benchmark.get(bench, {})
        ax.plot(agents, ratio, marker=marker_for_benchmark.get(bench, "o"),
                markersize=6, linewidth=sty.get("linewidth", 2),
                linestyle=sty.get("linestyle", "--"),
                color=sty.get("color", None), alpha=sty.get("alpha", 0.9),
                label=bench)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label=base_name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Speedup relative to NaiveFor (average across machines)")
    ax.set_xlabel("agents")
    ax.set_ylabel("time / time(NaiveFor)   (lower = faster)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "variantC_speedup_vs_naivefor.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant D -- heatmap (benchmark x agents) of mean time, one per family
# ---------------------------------------------------------------------------
def plot_variant_d(avg, out_dir):
    families = {
        "Naive": [b for b in avg if b.startswith("Naive")],
        "Binned": [b for b in avg if b.startswith("Binned")],
    }

    # determine the union of agent sizes used across all benchmarks
    all_agents = sorted({a for bd in avg.values() for a in bd})

    for family, benches in families.items():
        benches = sorted(benches)
        matrix = np.full((len(benches), len(all_agents)), np.nan)
        for i, bench in enumerate(benches):
            for j, a in enumerate(all_agents):
                if a in avg[bench]:
                    # fill with log10(mean time) so the colour scale is readable
                    matrix[i, j] = np.log10(avg[bench][a])

        fig, ax = plt.subplots(figsize=(10, 0.55 * len(benches) + 3))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(all_agents)))
        ax.set_xticklabels(all_agents)
        ax.set_yticks(range(len(benches)))
        ax.set_yticklabels(benches, fontsize=8)
        ax.set_xlabel("agents")
        ax.grid(False)
        for i in range(len(benches)):
            for j in range(len(all_agents)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                            fontsize=7, color="white")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log10(time_per_step_ms)")
        ax.set_title(f"{family} — mean time heatmap (average across machines)")
        fig.tight_layout()
        p = out_dir / f"variantD_heatmap_{family.lower()}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant E -- grouped bar chart (benchmark x agents), one panel per family
# ---------------------------------------------------------------------------
def plot_variant_e(avg, out_dir):
    families = {
        "Naive": [b for b in avg if b.startswith("Naive")],
        "Binned": [b for b in avg if b.startswith("Binned")],
    }
    all_agents = sorted({a for bd in avg.values() for a in bd})

    for family, benches in families.items():
        benches = sorted(benches)
        n_bench = len(benches)
        n_agents = len(all_agents)

        fig, ax = plt.subplots(figsize=(max(10, 1.4 * n_agents + 2),
                                        0.55 * n_bench + 3.5))
        width = 0.8 / n_bench
        positions = np.arange(n_agents)

        for i, bench in enumerate(benches):
            offsets = positions + (i - n_bench / 2) * width
            vals = [avg[bench].get(a, 0.0) for a in all_agents]
            sty = style_for_benchmark.get(bench, {})
            ax.bar(offsets, vals, width, label=bench,
                   color=sty.get("color", None), alpha=sty.get("alpha", 0.85))

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(all_agents)
        ax.set_xlabel("agents")
        ax.set_ylabel("time_per_step_ms (log)")
        ax.set_title(f"{family} — mean time per step (average across machines)")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        # keep the legend small and readable for so many series
        ax.legend(fontsize=7, ncol=2 if n_bench > 6 else 1)
        fig.tight_layout()
        p = out_dir / f"variantE_bars_{family.lower()}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant F -- family bands on ONE plot (crossover emphasised, no line overlap)
# ---------------------------------------------------------------------------
def _family_band(ax, avg, prefix, agents, color, label):
    """Draw the min/max band and the best/worst lines for one family."""
    benches = [b for b in avg if b.startswith(prefix)]
    series = {b: {a: avg[b][a] for a in agents if a in avg[b]} for b in benches}
    # only agents where at least one benchmark of the family has data
    usable = [a for a in agents if any(a in series[b] for b in series)]
    lo = [min(series[b][a] for b in series if a in series[b]) for a in usable]
    hi = [max(series[b][a] for b in series if a in series[b]) for a in usable]
    ax.fill_between(usable, lo, hi, color=color, alpha=0.15, label=None)
    ax.plot(usable, lo, color=color, linestyle="-", linewidth=2,
            label=f"{label} best")
    ax.plot(usable, hi, color=color, linestyle=":", linewidth=1.5,
            label=f"{label} worst")


def plot_variant_f(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(11, 7))
    _family_band(ax, avg, "Naive", agents, "tab:blue", "Naive")
    _family_band(ax, avg, "Binned", agents, "tab:orange", "Binned")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Naive vs Binned families \u2014 spread shown as bands\n"
                 "(crossover = where lines cross)")
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "variantF_family_bands.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant G -- naive vs binned scatter (same plot, speedup/crossover, no overlap)
# ---------------------------------------------------------------------------
# mapping from a naive benchmark to its binned counterpart (same access pattern)
NAIVE_TO_BINNED = {
    "NaiveFor": "BinnedReduceFor",
    "NaiveForDouble": "BinnedReduceForDouble",
    "NaiveForSymmetric": "BinnedReduceForSymmetric",
    "NaiveParallelReduce": "BinnedReduceParallel",
    "NaiveParallelReduceDouble": "BinnedReduceParallelDouble",
    "NaiveParallelReduceSymmetric": "BinnedReduceParallelSymmetric",
}


def plot_variant_g(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("viridis")
    norm_agents = np.log2(max(agents) / min(agents))

    for naive_name, binned_name in NAIVE_TO_BINNED.items():
        if naive_name not in avg or binned_name not in avg:
            continue
        xs, ys, cs = [], [], []
        for a in agents:
            if a in avg[naive_name] and a in avg[binned_name]:
                xs.append(avg[naive_name][a])   # naive time
                ys.append(avg[binned_name][a])  # binned time
                cs.append(np.log2(a / min(agents)) / norm_agents)
        ax.scatter(xs, ys, c=cs, cmap=cmap, s=40, alpha=0.9,
                   label=naive_name.replace("Naive", ""))

    # equality line: above = naive faster (below = binned faster)
    allx = [v for bd in avg.values() for v in bd.values()]
    lo, hi = min(allx), max(allx)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="equal time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("naive time_per_step_ms")
    ax.set_ylabel("binned time_per_step_ms")
    ax.set_title("Naive vs Binned per access pattern\n"
                 "(above diagonal = naive faster, below = binned faster;\n"
                 "colour = agents, dark = large)")
    ax.grid(True, which="both", alpha=0.3)
    sc = ax.collections[0]
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2(agents)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = out_dir / "variantG_naive_vs_binned_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant H -- best-of-family lines on ONE plot (cleanest crossover view)
# ---------------------------------------------------------------------------
def _best_line(avg, prefix, agents):
    benches = [b for b in avg if b.startswith(prefix)]
    usable = [a for a in agents if any(a in avg[b] for b in benches)]
    best = [min(avg[b][a] for b in benches if a in avg[b]) for a in usable]
    return usable, best


def plot_variant_h(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(11, 7))
    aN, bestN = _best_line(avg, "Naive", agents)
    aB, bestB = _best_line(avg, "Binned", agents)
    ax.plot(aN, bestN, color="tab:blue", linewidth=3, marker="o", markersize=5,
            label="Naive (best of 7)")
    ax.plot(aB, bestB, color="tab:orange", linewidth=3, marker="s", markersize=5,
            label="Binned (best of 6)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Best Naive vs Best Binned \u2014 the crossover")
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "variantH_best_of_family.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant I -- one panel per access pattern (For / Parallel / Spread)
# ---------------------------------------------------------------------------
def plot_variant_i(avg, out_dir):
    groups = {
        "For": [b for b in avg if "For" in b],
        "Parallel": [b for b in avg if "Parallel" in b],
        "Spread": [b for b in avg if "Spread" in b],
    }
    all_agents = sorted({a for bd in avg.values() for a in bd})
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)

    for ax, (name, benches) in zip(axes, groups.items()):
        for bench in sorted(benches):
            sty = style_for_benchmark.get(bench, {})
            pts = [(a, avg[bench][a]) for a in all_agents if a in avg[bench]]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker=marker_for_benchmark.get(bench, "o"), markersize=6,
                    linewidth=sty.get("linewidth", 2),
                    linestyle=sty.get("linestyle", "--"),
                    color=sty.get("color", None), alpha=sty.get("alpha", 0.85),
                    label=bench)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("agents")
        ax.set_ylabel("time_per_step_ms")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Grouped by access pattern \u2014 fewer overlapping lines per panel", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "variantI_grouped_by_pattern.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant J -- speedup-ratio lines (binned/naive) vs agents; crossover at 1
# ---------------------------------------------------------------------------
def plot_variant_j(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(11, 7))
    for naive_name, binned_name in NAIVE_TO_BINNED.items():
        if naive_name not in avg or binned_name not in avg:
            continue
        pts = [(a, avg[binned_name][a] / avg[naive_name][a]) for a in agents
               if a in avg[naive_name] and a in avg[binned_name]]
        pts.sort()
        sty = style_for_benchmark.get(naive_name, {})
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker_for_benchmark.get(naive_name, "o"), markersize=5,
                linewidth=1.6, linestyle=sty.get("linestyle", "--"),
                color=sty.get("color", None), alpha=0.9,
                label=naive_name.replace("Naive", ""))
    ax.axhline(1.0, color="black", linestyle="-", linewidth=1.5,
               label="equal (ratio = 1)")
    # shade the region where binned wins
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_xlabel("agents")
    ax.set_ylabel("binned / naive time  (below 1 = binned faster)")
    ax.set_title("Speedup: binned / naive vs agents \u2014 crossover at ratio 1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "variantJ_speedup_ratio.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant K -- box plot of each family per agent count (no individual lines)
# ---------------------------------------------------------------------------
def plot_variant_k(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    families = {"Naive": [b for b in avg if b.startswith("Naive")],
                "Binned": [b for b in avg if b.startswith("Binned")]}

    n = len(agents)
    positions = []
    boxes_naive, boxes_binned = [], []
    ticks = []
    for j, a in enumerate(agents):
        pos_naive = j * 2 - 0.25
        pos_binned = j * 2 + 0.25
        positions += [pos_naive, pos_binned]
        vn = [avg[b][a] for b in families["Naive"] if a in avg[b]]
        vb = [avg[b][a] for b in families["Binned"] if a in avg[b]]
        boxes_naive.append(vn)
        boxes_binned.append(vb)
        ticks.append(j * 2)

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * n + 3), 7))
    ax.boxplot(boxes_naive, positions=positions[0::2], widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.5))
    ax.boxplot(boxes_binned, positions=positions[1::2], widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="tab:orange", alpha=0.5))
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(agents, rotation=60, fontsize=7)
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms (log)")
    ax.set_title("Family distribution per agent count \u2014 blue = Naive, orange = Binned")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="tab:blue", alpha=0.5, label="Naive (7)"),
                       Patch(facecolor="tab:orange", alpha=0.5, label="Binned (6)")],
              fontsize=9)
    fig.tight_layout()
    p = out_dir / "variantK_family_boxplot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant L -- all 13 lines subdued, best naive / best binned emphasised
# ---------------------------------------------------------------------------
def plot_variant_l(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(11, 7))
    # draw every benchmark faintly so the full data is present
    for bench, bench_data in sorted(avg.items()):
        pts = [(a, bench_data[a]) for a in agents if a in bench_data]
        pts.sort()
        sty = style_for_benchmark.get(bench, {})
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker_for_benchmark.get(bench, "."), markersize=3,
                linewidth=0.8, linestyle=sty.get("linestyle", "--"),
                color=sty.get("color", None), alpha=0.25)
    # emphasise the best-of-family lines on top
    aN, bestN = _best_line(avg, "Naive", agents)
    aB, bestB = _best_line(avg, "Binned", agents)
    ax.plot(aN, bestN, color="tab:blue", linewidth=3.5, marker="o", markersize=6,
            label="Naive best")
    ax.plot(aB, bestB, color="tab:orange", linewidth=3.5, marker="s", markersize=6,
            label="Binned best")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("All variants (faint) + best-of-family emphasised")
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "variantL_all_subdued_best_emphasised.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant M -- parallel slope chart: two vertical axes, one wrap per variant
# ---------------------------------------------------------------------------
def plot_variant_m(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    # average over agents so each variant is one line from naive-axis to binned-axis
    pair_rows = []
    for naive_name, binned_name in NAIVE_TO_BINNED.items():
        if naive_name not in avg or binned_name not in avg:
            continue
        # pick a representative mid-large agent count where the crossover is visible
        big = [a for a in agents if a >= 2048 and a in avg[naive_name]
               and a in avg[binned_name]]
        a = big[-1] if big else agents[-1]
        pair_rows.append((naive_name, avg[naive_name][a], avg[binned_name][a]))

    styles = {b: style_for_benchmark.get(b, {}) for b, _, _ in pair_rows}
    colors = [styles[b].get("color", "gray") for b, _, _ in pair_rows]
    linestyles = [styles[b].get("linestyle", "--") for b, _, _ in pair_rows]

    fig, ax = plt.subplots(figsize=(9, 8))
    # two vertical axes positions
    x_naive, x_binned = 1.0, 2.0
    yvals_naive = [r[1] for r in pair_rows]
    yvals_binned = [r[2] for r in pair_rows]

    # scale both axes to the same relative range so slopes are comparable
    allv = yvals_naive + yvals_binned
    lo, hi = min(allv), max(allv)

    labels = [r[0].replace("Naive", "") for r in pair_rows]
    for i, (b, yn, yb) in enumerate(pair_rows):
        ax.plot([x_naive, x_binned], [yn, yb], color=colors[i],
                linestyle=linestyles[i], linewidth=2, alpha=0.85,
                marker="o", markersize=6)

    # label each right-hand end
    for i, (b, yn, yb) in enumerate(pair_rows):
        ax.annotate(labels[i], (x_binned, yb), xytext=(6, 0),
                    textcoords="offset points", fontsize=8,
                    color=colors[i], va="center")

    ax.set_xticks([x_naive, x_binned])
    ax.set_xticklabels(["Naive", "Binned"], fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(lo / 2, hi * 2)
    ax.set_title(f"Parallel slope @ agents={a}\n"
                 "(steeper up-slope = binned slower there; \n"
                 "down-slope = binned faster)")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantM_parallel_slope.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant N -- 100% stacked area: family share of runtime vs agents
# ---------------------------------------------------------------------------
def plot_variant_n(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})

    def family_total(prefix, a):
        vals = [avg[b][a] for b in avg if b.startswith(prefix) and a in avg[b]]
        return sum(vals) if vals else 0.0

    n_tot = [family_total("Naive", a) for a in agents]
    b_tot = [family_total("Binned", a) for a in agents]

    # normalise to fractions of total runtime across BOTH families
    frac_n = np.array(n_tot)
    frac_b = np.array(b_tot)
    total = frac_n + frac_b
    frac_n = frac_n / total
    frac_b = frac_b / total

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(agents, frac_n, frac_b,
                 labels=["Naive (7)", "Binned (6)"],
                 colors=["tab:blue", "tab:orange"], alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total runtime")
    ax.set_xlabel("agents")
    ax.set_title("Runtime share \u2014 when does Binned start dominating the cost?")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    p = out_dir / "variantN_stacked_area.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant O -- divergence filled area: log10(binned/naive) above/below zero
# ---------------------------------------------------------------------------
def plot_variant_o(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(11, 7))
    for naive_name, binned_name in NAIVE_TO_BINNED.items():
        if naive_name not in avg or binned_name not in avg:
            continue
        pts = [(a, np.log10(avg[binned_name][a] / avg[naive_name][a])) for a in agents
               if a in avg[naive_name] and a in avg[binned_name]]
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        sty = style_for_benchmark.get(naive_name, {})
        ax.fill_between(xs, ys, 0, color=sty.get("color", None),
                        alpha=0.25)
        ax.plot(xs, ys, color=sty.get("color", None),
                linestyle=sty.get("linestyle", "--"), linewidth=1.6,
                label=naive_name.replace("Naive", ""))
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_ylabel("log10(binned / naive)\n(negative = binned faster)")
    ax.set_xlabel("agents")
    ax.set_title("Divergence \u2014 where the speedup flips sign")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    p = out_dir / "variantO_divergence_filled.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant P -- single speedup-ratio heatmap: variant x agents, colour = ratio
# ---------------------------------------------------------------------------
def plot_variant_p(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    pairs = list(NAIVE_TO_BINNED.items())
    labels = [n.replace("Naive", "") for n, _ in pairs]

    matrix = np.full((len(pairs), len(agents)), np.nan)
    for i, (n, b) in enumerate(pairs):
        for j, a in enumerate(agents):
            if a in avg[n] and a in avg[b]:
                # log speedup: positive = binned faster
                matrix[i, j] = np.log10(avg[n][a] / avg[b][a])

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(pairs) + 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(matrix)), vmax=np.nanmax(np.abs(matrix)))
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, fontsize=7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("agents")
    for i in range(len(pairs)):
        for j in range(len(agents)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.5 * np.nanmax(
                            np.abs(matrix)) else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 speedup (positive = binned faster)")
    ax.set_title("Binned vs Naive speedup hot/cold heatmap")
    fig.tight_layout()
    p = out_dir / "variantP_speedup_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant S -- diverging waterfall bars: best binned vs best naive per agents
# ---------------------------------------------------------------------------
def plot_variant_s(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    aN, bestN = _best_line(avg, "Naive", agents)
    aB, bestB = _best_line(avg, "Binned", agents)

    # log speedup of binned relative to naive (positive = binned faster)
    lookupN = dict(zip(aN, bestN))
    lookupB = dict(zip(aB, bestB))
    common = [a for a in agents if a in lookupN and a in lookupB]
    speeds = [np.log10(lookupN[a] / lookupB[a]) for a in common]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(common) + 3), 7))
    colors = ["tab:orange" if s > 0 else "tab:blue" for s in speeds]
    ax.bar(range(len(common)), speeds, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(common, rotation=60, fontsize=7)
    ax.set_xlabel("agents")
    ax.set_ylabel("log10(best Naive / best Binned)\n(+ = binned faster, - = naive faster)")
    ax.set_title("Best Binned vs Best Naive \u2014 diverging speedup bar")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantS_diverging_waterfall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant T -- range / bullet chart: per-agent spread band, variants marked
# ---------------------------------------------------------------------------
def plot_variant_t(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(13, 7))

    for j, a in enumerate(agents):
        # per-family ranges for this agent count
        nvals = {b: avg[b][a] for b in avg if b.startswith("Naive") and a in avg[b]}
        bvals = {b: avg[b][a] for b in avg if b.startswith("Binned") and a in avg[b]}
        if not nvals or not bvals:
            continue
        # x position (two lanes per agent count)
        xN = j * 3 - 0.6
        xB = j * 3 + 0.6

        # range band (worst .. best) as a thick line
        ax.plot([xN, xN], [min(nvals.values()), max(nvals.values())],
                color="tab:blue", linewidth=8, alpha=0.35, solid_capstyle="round")
        ax.plot([xB, xB], [min(bvals.values()), max(bvals.values())],
                color="tab:orange", linewidth=8, alpha=0.35, solid_capstyle="round")

        # mark every individual variant as a small dot inside its band
        for v in nvals.values():
            ax.plot(xN, v, "o", color="tab:blue", markersize=3)
        for v in bvals.values():
            ax.plot(xB, v, "o", color="tab:orange", markersize=3)

        # emphasise the best of each family
        ax.plot(xN, min(nvals.values()), "o", color="white", mec="tab:blue",
                markersize=6, zorder=5)
        ax.plot(xB, min(bvals.values()), "o", color="white", mec="tab:orange",
                markersize=6, zorder=5)

    ax.set_yscale("log")
    ax.set_xticks([j * 3 for j in range(len(agents))])
    ax.set_xticklabels(agents, rotation=60, fontsize=7)
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.set_title("Range chart \u2014 Naive (blue) vs Binned (orange) spread per size;\n"
                 "band = worst-to-best, dots = each variant, white ring = best")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantT_range_chart.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant U -- radar chart: Naive vs Binned across access patterns
# ---------------------------------------------------------------------------
def plot_variant_u(avg, out_dir):
    pairs = list(NAIVE_TO_BINNED.items())
    labels = [n.replace("Naive", "").replace("Parallel", "Par.") for n, _ in pairs]

    # choose an agent count where the contrast is informative (mid crossover)
    agents = sorted({a for bd in avg.values() for a in bd})
    target = [a for a in agents if a >= 4096]
    a = target[0] if target else agents[-1]

    n_pairs = len(pairs)
    angles = np.linspace(0, 2 * np.pi, n_pairs, endpoint=False).tolist() + [0.0]

    def series_for(prefix):
        vals = []
        for n, b in pairs:
            src = avg[n] if prefix == "Naive" else avg[b]
            vals.append(src[a] if a in src else np.nan)
        # normalise each axis to 1 = best (fastest) among the two families
        norm = []
        for (n, b) in pairs:
            mn = min(avg[n][a], avg[b][a]) if a in avg[n] and a in avg[b] else np.nan
            v = avg[n][a] if prefix == "Naive" else avg[b][a]
            norm.append(mn / v if mn and v else np.nan)  # 1 = fastest, <1 slower
        return norm + [norm[0]]

    norm_naive = series_for("Naive")
    norm_binned = series_for("Binned")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_naive, color="tab:blue", linewidth=2, label="Naive")
    ax.fill(angles, norm_naive, color="tab:blue", alpha=0.2)
    ax.plot(angles, norm_binned, color="tab:orange", linewidth=2, label="Binned")
    ax.fill(angles, norm_binned, color="tab:orange", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Radar @ agents={a}\n(1 = fastest on that axis; farther out = faster)",
                 fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    fig.tight_layout()
    p = out_dir / "variantU_radar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant V -- stacked area chart (non-normalised) of all variants
# ---------------------------------------------------------------------------
def plot_variant_v(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    # order: binned first (they dominate at large N), then naive
    benches = [b for b in avg if b.startswith("Binned")] + \
              [b for b in avg if b.startswith("Naive")]

    # build a matrix, filling missing agent counts with the previous value
    values = np.zeros((len(benches), len(agents)))
    for i, b in enumerate(benches):
        last = 0.0
        for j, a in enumerate(agents):
            last = avg[b].get(a, last)
            values[i, j] = last

    colors = [style_for_benchmark.get(b, {}).get("color", "gray") for b in benches]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(agents, values, labels=benches, colors=colors, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("agents")
    ax.set_ylabel("cumulative time_per_step_ms")
    ax.set_title("Stacked area of all variants (Binned layers base)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantV_stacked_area.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant W -- stream graph (symmetric stacked area)
# ---------------------------------------------------------------------------
def plot_variant_w(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    benches = [b for b in avg if b.startswith("Binned")] + \
              [b for b in avg if b.startswith("Naive")]

    values = np.zeros((len(benches), len(agents)))
    for i, b in enumerate(benches):
        last = 0.0
        for j, a in enumerate(agents):
            last = avg[b].get(a, last)
            values[i, j] = last

    # scale each variant by its maximum so they stay comparable in a stream
    maxes = values.max(axis=1, keepdims=True)
    maxes[maxes == 0] = 1.0
    scaled = values / maxes

    colors = [style_for_benchmark.get(b, {}).get("color", "gray") for b in benches]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(agents, scaled, labels=benches, colors=colors, alpha=0.85,
                 baseline="wiggle")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("agents")
    ax.set_ylabel("normalised runtime (each variant scaled to its own max)")
    ax.set_title("Stream graph of all variants\n(ribbon width = relative runtime vs its own peak)")
    ax.legend(fontsize=5, ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    p = out_dir / "variantW_stream_graph.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant X -- gantt / range bars: one horizontal bar per variant
# ---------------------------------------------------------------------------
def plot_variant_x(avg, out_dir):
    benches = [b for b in avg if b.startswith("Naive")] + \
              [b for b in avg if b.startswith("Binned")]

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(benches) + 2))
    for i, b in enumerate(benches):
        vals = avg[b]
        lo, hi = min(vals.values()), max(vals.values())
        color = style_for_benchmark.get(b, {}).get("color", "gray")
        ax.barh(i, hi - lo, left=lo, height=0.6, color=color, alpha=0.6)
        ax.plot(lo, i, "o", color="white", mec=color, markersize=6)
        ax.plot(hi, i, "o", color="white", mec=color, markersize=6)
        ax.text(hi, i, f" {lo:.2g}", va="center", ha="left", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(range(len(benches)))
    ax.set_yticklabels(benches, fontsize=8)
    ax.set_xlabel("time_per_step_ms (range min\u2192max over agent sizes)")
    ax.set_title("Gantt/range \u2014 each variant\u2019s time corridor across agent sizes")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantX_gantt_range.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant Y -- diverging chart with BOTH families, non-extreme variants marked
# ---------------------------------------------------------------------------
def plot_variant_y(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(13, 8))

    width = 0.8
    for a in agents:
        nvals = {b: avg[b][a] for b in avg if b.startswith("Naive") and a in avg[b]}
        bvals = {b: avg[b][a] for b in avg if b.startswith("Binned") and a in avg[b]}
        if not nvals or not bvals:
            continue
        # reference = the overall fastest (best) across both families
        ref = min([min(nvals.values()), min(bvals.values())])
        # log-slowdown factor relative to the family whose best we highlight

        # Naive lane (left half of the slot)
        bestN = min(nvals.values())
        worstN = max(nvals.values())
        # bar: range from best to worst relative to the global best
        n_lo = np.log10(bestN / ref)
        n_hi = np.log10(worstN / ref)
        ax.barh(0, n_hi - n_lo, left=n_lo, height=width * 0.9,
                color="tab:blue", alpha=0.5, align="edge")
        # mark all variants (dots), with best ringed
        for b, v in nvals.items():
            x = np.log10(v / ref)
            is_best = (v == bestN)
            ax.plot(x, 0.45, "o", color="tab:blue" if not is_best else "white",
                    mec="tab:blue", markersize=5 if not is_best else 8, zorder=5)

        # Binned lane (right half of the slot)
        bestB = min(bvals.values())
        worstB = max(bvals.values())
        b_lo = np.log10(bestB / ref)
        b_hi = np.log10(worstB / ref)
        ax.barh(1, b_hi - b_lo, left=b_lo, height=width * 0.9,
                color="tab:orange", alpha=0.5, align="edge")
        for b, v in bvals.items():
            x = np.log10(v / ref)
            is_best = (v == bestB)
            ax.plot(x, 1.45, "o", color="tab:orange" if not is_best else "white",
                    mec="tab:orange", markersize=5 if not is_best else 8, zorder=5)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks([0.45, 1.45])
    ax.set_yticklabels(["Naive", "Binned"])
    ax.set_xlabel("log10(slowdown vs overall fastest at that size)\n(0 = fastest, larger = slower)")
    ax.set_title("Both families per size \u2014 bars = worst-to-best range,\n"
                 "dots = every variant, white ring = that family\u2019s best")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "variantY_both_families_diverging.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant Z -- family bands with all benchmarks drawn faintly inside,
#              best & worst labelled by name on the plot
# ---------------------------------------------------------------------------
def _best_worst_names(avg, prefix, agents):
    """Return (bench_best, bench_worst, mid_agent) for a family.

    The best/worst name is determined by the median slowdown over the
    agent range (robust to the few stray points at the extremes).
    """
    benches = [b for b in avg if b.startswith(prefix)]
    usable = [a for a in agents if any(a in avg[b] for b in benches)]
    mid = usable[len(usable) // 2]
    scores = {}
    for b in benches:
        # mean of log10 time across the covered range (log is scale-invariant)
        covered = [a for a in usable if a in avg[b]]
        if covered:
            scores[b] = np.mean([np.log10(avg[b][a]) for a in covered])
    bench_best = min(scores, key=scores.get)
    bench_worst = max(scores, key=scores.get)
    return bench_best, bench_worst, mid


def plot_variant_z(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(13, 8))

    for prefix, color in (("Naive", "tab:blue"), ("Binned", "tab:orange")):
        benches = [b for b in avg if b.startswith(prefix)]
        series = {b: {a: avg[b][a] for a in agents if a in avg[b]} for b in benches}
        usable = [a for a in agents if any(a in series[b] for b in series)]
        lo = [min(series[b][a] for b in series if a in series[b]) for a in usable]
        hi = [max(series[b][a] for b in series if a in series[b]) for a in usable]

        # band between worst and best
        ax.fill_between(usable, lo, hi, color=color, alpha=0.12, label=None)
        ax.plot(usable, lo, color=color, linestyle="-", linewidth=2.5,
                label=f"{prefix} best")
        ax.plot(usable, hi, color=color, linestyle=":", linewidth=2,
                label=f"{prefix} worst")

        # every benchmark drawn faintly inside the band
        for b, s in series.items():
            pts = [(a, s[a]) for a in usable if a in s]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linewidth=1.0, alpha=0.30, zorder=3)

        # identify best & worst by name and annotate their lines
        bench_best, bench_worst, _ = _best_worst_names(avg, prefix, agents)
        # re-draw the best/worst benchmark lines with distinct emphasis
        bb = [(a, avg[bench_best][a]) for a in usable if a in avg[bench_best]]
        bw = [(a, avg[bench_worst][a]) for a in usable if a in avg[bench_worst]]
        ax.plot([p[0] for p in bb], [p[1] for p in bb], color=color,
                linewidth=3.0, linestyle="-", alpha=1.0, zorder=6,
                label=f"{bench_best}")
        ax.plot([p[0] for p in bw], [p[1] for p in bw], color=color,
                linewidth=2.5, linestyle=":", alpha=0.9, zorder=6,
                label=f"{bench_worst}")
        # name labels at the right edge (mid agent)
        if bb:
            ax.annotate(bench_best, (bb[-1][0], bb[-1][1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color=color, va="center")
        if bw:
            ax.annotate(bench_worst, (bw[-1][0], bw[-1][1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=9, fontstyle="italic",
                        color=color, va="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Family bands with all benchmarks \u2014\n"
                 "band = worst-to-best, faint lines = every benchmark,\n"
                 "bold = best & italic = worst, named on the plot")
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out_dir / "variantZ_bands_with_all_benchmarks.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant AA -- small multiples: ALL benchmarks, one clean panel each, shared
#               axes so magnitudes are directly comparable, zero overlap.
# ---------------------------------------------------------------------------
def plot_variant_aa(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})

    # order all 13 benchmarks: group by access pattern (For / Parallel / Spread),
    # keeping Naive before Binned within each group so the speedup pairing reads.
    def pattern_key(b):
        if "Parallel" in b:
            return (1, b.replace("NaiveParallel", "").replace("BinnedReduceParallel", ""))
        if "Spread" in b:
            return (2, b)
        return (0, b.replace("Naive", "").replace("BinnedReduce", ""))

    benches = sorted(avg.keys(), key=pattern_key)
    n = len(benches)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    # shared axis limits (so panels are comparable to each other)
    all_vals = [v for bd in avg.values() for v in bd.values()]
    ylo, yhi = min(all_vals), max(all_vals)
    xlo, xhi = min(agents), max(agents)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    family_color = {"Naive": "tab:blue", "Binned": "tab:orange"}

    for i, axes2 in enumerate(axes):
        if i >= n:
            axes2.set_visible(False)
            continue
        bench = benches[i]
        prefix = "Naive" if bench.startswith("Naive") else "Binned"
        color = family_color[prefix]

        data = [(a, avg[bench][a]) for a in agents if a in avg[bench]]
        data.sort()
        axes2.plot([p[0] for p in data], [p[1] for p in data],
                   color=color, linewidth=2.2, marker="o", markersize=3)
        axes2.set_title(bench, fontsize=9, color=color, fontweight="bold")
        axes2.set_xscale("log", base=2)
        axes2.set_yscale("log")
        axes2.set_xlim(xlo, xhi)
        axes2.set_ylim(ylo, yhi)
        axes2.grid(True, which="both", alpha=0.25)
        # tint each panel's spine by family so the grouping reads at a glance
        for spine in axes2.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.2)

    # enable x labels on the bottom-most occupied row, y labels on the left column
    last_row = (n - 1) // ncols
    for i, ax in enumerate(axes):
        if i >= n:
            break
        row, col = divmod(i, ncols)
        ax.tick_params(labelbottom=(row == last_row), labelleft=(col == 0),
                       labelsize=7)

    fig.suptitle("All 13 benchmarks \u2014 small multiples (shared axes)\n"
                 "blue = Naive, orange = Binned; each panel same scale \u2014 no overlap",
                 fontsize=12)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="tab:blue", label="Naive (7)"),
                        Patch(facecolor="tab:orange", label="Binned (6)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = out_dir / "variantAA_small_multiples_all.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# Variant AB -- ONE single plot with ALL 13 benchmarks, styled lines, plus the
#               two family bands underneath to emphasise the speedup envelope.
# ---------------------------------------------------------------------------
def plot_variant_ab(avg, out_dir):
    agents = sorted({a for bd in avg.values() for a in bd})
    fig, ax = plt.subplots(figsize=(13, 8))

    # 1) family bands underneath (speedup envelope)
    for prefix, color in (("Naive", "tab:blue"), ("Binned", "tab:orange")):
        benches = [b for b in avg if b.startswith(prefix)]
        series = {b: {a: avg[b][a] for a in agents if a in avg[b]} for b in benches}
        usable = [a for a in agents if any(a in series[b] for b in series)]
        lo = [min(series[b][a] for b in series if a in series[b]) for a in usable]
        hi = [max(series[b][a] for b in series if a in series[b]) for a in usable]
        ax.fill_between(usable, lo, hi, color=color, alpha=0.08)
        ax.plot(usable, lo, color=color, linewidth=1.8, alpha=0.6, zorder=2)
        ax.plot(usable, hi, color=color, linewidth=1.8, alpha=0.6, zorder=2,
                linestyle=":")

    # 2) every benchmark as a distinct styled + coloured line on top.
    # Distinct colour per benchmark, GROUPED by family:
    # Naive = a tight family of blue/cyan/teal shades; Binned = a tight family
    # of red/orange/magenta shades.  The two families are separated by a hard
    # hue boundary so you can tell family at a glance, while each line's shade
    # (plus linestyle/marker) still separates it from its siblings.
    naive_colors = {
        "NaiveFor": "#0b3d91",                # deep navy
        "NaiveForDouble": "#1f77b4",          # medium blue
        "NaiveForSymmetric": "#58a6ff",       # light blue
        "NaiveParallelReduce": "#17becf",     # cyan
        "NaiveParallelReduceDouble": "#00a6a6",  # dark cyan
        "NaiveParallelReduceSymmetric": "#20b2aa",  # light teal
        "NaiveSpread": "#006d77",             # sea teal
    }
    binned_colors = {
        "BinnedReduceFor": "#8b0000",         # dark red
        "BinnedReduceForDouble": "#d62728",   # red
        "BinnedReduceForSymmetric": "#e85d04",  # red-orange
        "BinnedReduceParallel": "#ff7f0e",    # orange
        "BinnedReduceParallelDouble": "#ffa500",  # amber orange
        "BinnedReduceParallelSymmetric": "#c9184a",  # raspberry/magenta-red
    }
    # per-family linestyles so close lines with the same colour family still separate
    naive_ls = {   b: ls for b, ls in zip(
        ["NaiveFor", "NaiveForDouble", "NaiveForSymmetric",
         "NaiveParallelReduce", "NaiveParallelReduceDouble",
         "NaiveParallelReduceSymmetric", "NaiveSpread"],
        ["-", "--", "-.", "-", "--", "-.", ":"])
    }
    binned_ls = { b: ls for b, ls in zip(
        ["BinnedReduceFor", "BinnedReduceForDouble", "BinnedReduceForSymmetric",
         "BinnedReduceParallel", "BinnedReduceParallelDouble",
         "BinnedReduceParallelSymmetric"],
        ["-", "--", "-.", "-", "--", "-."])
    }

    for bench, bench_data in sorted(avg.items()):
        prefix = "Naive" if bench.startswith("Naive") else "Binned"
        color = naive_colors.get(bench) or binned_colors.get(bench)
        linestyle = (naive_ls if prefix == "Naive" else binned_ls).get(bench, "-")
        marker = marker_for_benchmark.get(bench, "o")
        pts = [(a, bench_data[a]) for a in agents if a in bench_data]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, linestyle=linestyle, marker=marker, markersize=4,
                linewidth=1.6, alpha=0.9, zorder=4, label=bench)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("All 13 benchmarks on one plot\n"
                 "shaded = Naive / Binned worst-to-best envelope (the speedup gap),\n"
                 "distinct colour per benchmark \\u2014 cool = Naive, warm = Binned")
    ax.set_xlabel("agents")
    ax.set_ylabel("time_per_step_ms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    p = out_dir / "variantAB_single_plot_all_benchmarks.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results.csv"
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    rows = load_rows(csv_path)
    avg = average_over_machines(rows)

    plot_variant_a(avg, out_dir)
    plot_variant_b(rows, out_dir)
    plot_variant_c(avg, out_dir)
    plot_variant_d(avg, out_dir)
    plot_variant_e(avg, out_dir)
    plot_variant_f(avg, out_dir)
    plot_variant_g(avg, out_dir)
    plot_variant_h(avg, out_dir)
    plot_variant_i(avg, out_dir)
    plot_variant_j(avg, out_dir)
    plot_variant_k(avg, out_dir)
    plot_variant_l(avg, out_dir)
    plot_variant_m(avg, out_dir)
    plot_variant_n(avg, out_dir)
    plot_variant_o(avg, out_dir)
    plot_variant_p(avg, out_dir)
    plot_variant_s(avg, out_dir)
    plot_variant_t(avg, out_dir)
    plot_variant_u(avg, out_dir)
    plot_variant_v(avg, out_dir)
    plot_variant_w(avg, out_dir)
    plot_variant_x(avg, out_dir)
    plot_variant_y(avg, out_dir)
    plot_variant_z(avg, out_dir)
    plot_variant_aa(avg, out_dir)
    plot_variant_ab(avg, out_dir)


if __name__ == "__main__":
    main()
