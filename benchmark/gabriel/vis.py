import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

marker_for_benchmark = {
    "NaiveFor":                             "o",
    "NaiveForDouble":                       "v",
    "NaiveForSymmetric":                    "<",
    "NaiveParallelReduce":                  "s",
    "NaiveParallelReduceDouble":            "P",
    "NaiveParallelReduceSymmetric":         "X",
    "NaiveSpread":                          "^",
    "BinnedGabrielReduceFor":               "D",
    "BinnedGabrielReduceForDouble":         "d",
    "BinnedGabrielReduceForSymmetric":      "p",
    "BinnedGabrielReduceParallel":          "h",
    "BinnedGabrielReduceParallelDouble":    "H",
    "BinnedGabrielReduceParallelSymmetric": "*",
}

style_for_benchmark = {
    "NaiveFor":                             {"linestyle": "-",  "color": "tab:blue",   "alpha": 0.5, "linewidth": 1.5},
    "NaiveForDouble":                       {"linestyle": "--", "color": "tab:blue",   "alpha": 0.7, "linewidth": 2.0},
    "NaiveForSymmetric":                    {"linestyle": ":",  "color": "tab:blue",   "alpha": 1.0, "linewidth": 2.5},
    "NaiveParallelReduce":                  {"linestyle": "-",  "color": "tab:orange", "alpha": 0.5, "linewidth": 1.5},
    "NaiveParallelReduceDouble":            {"linestyle": "--", "color": "tab:orange", "alpha": 0.7, "linewidth": 2.0},
    "NaiveParallelReduceSymmetric":         {"linestyle": ":",  "color": "tab:orange", "alpha": 1.0, "linewidth": 2.5},
    "NaiveSpread":                          {"linestyle": "-",  "color": "tab:green",  "alpha": 0.5, "linewidth": 1.5},
    "BinnedGabrielReduceFor":               {"linestyle": "-",  "color": "tab:purple", "alpha": 0.5, "linewidth": 1.5},
    "BinnedGabrielReduceForDouble":         {"linestyle": "--", "color": "tab:purple", "alpha": 0.7, "linewidth": 2.0},
    "BinnedGabrielReduceForSymmetric":      {"linestyle": ":",  "color": "tab:purple", "alpha": 1.0, "linewidth": 2.5},
    "BinnedGabrielReduceParallel":          {"linestyle": "-",  "color": "tab:brown", "alpha": 0.5, "linewidth": 1.5},
    "BinnedGabrielReduceParallelDouble":    {"linestyle": "--", "color": "tab:brown", "alpha": 0.7, "linewidth": 2.0},
    "BinnedGabrielReduceParallelSymmetric": {"linestyle": ":",  "color": "tab:brown", "alpha": 1.0, "linewidth": 2.5},
}

def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results.csv"
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    rows = load_rows(csv_path)

    cleaned = []
    for r in rows:
        if not r.get("machine"):
            continue
        agents_str = r.get("agents")
        if not agents_str:
            continue
        try:
            agents_int = int(agents_str)
            time_val = float(r.get("time_per_step_ms") or 0.0)
        except (ValueError, TypeError):
            continue
        r["_agents_int"] = agents_int
        r["_time"] = time_val
        cleaned.append(r)
    rows = cleaned

    # averages over all machines
    avg_acc = defaultdict(list)
    for r in rows:
        key = (r["benchmark"], r["_agents_int"])
        avg_acc[key].append(r["_time"])

    avg_by_benchmark = defaultdict(list)
    for (bench, agents), vals in avg_acc.items():
        avg_by_benchmark[bench].append((agents, mean(vals)))

    def plot_benchmarks(ax, data, title):
        """Plot benchmark data onto an axis.  `data` is {benchmark: [(agents, time), ...]}."""
        for benchmark, pairs in sorted(data.items()):
            pairs.sort(key=lambda p: p[0])
            agents = [p[0] for p in pairs]
            times = [p[1] for p in pairs]
            marker = marker_for_benchmark.get(benchmark, "o")
            sty = style_for_benchmark.get(benchmark, {})
            ax.plot(
                agents, times,
                marker=marker, markersize=7,
                linewidth=sty.get("linewidth", 2),
                linestyle=sty.get("linestyle", "--"),
                color=sty.get("color", None),
                alpha=sty.get("alpha", 0.8),
                label=benchmark,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("agents")
        ax.set_ylabel("time_per_step_ms")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # --- per-machine plots ---
    by_machine = defaultdict(list)
    for row in rows:
        if not row.get("machine"):
            continue
        by_machine[row["machine"]].append(row)

    for machine, machine_rows in by_machine.items():
        by_benchmark = defaultdict(list)
        for row in machine_rows:
            by_benchmark[row["benchmark"]].append((row["_agents_int"], row["_time"]))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_benchmarks(ax, by_benchmark, machine)
        fig.tight_layout()
        out_path = out_dir / f"{machine}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")

    # --- average plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_benchmarks(ax, avg_by_benchmark, "Average across machines")
    fig.tight_layout()
    out_path = out_dir / "average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()
