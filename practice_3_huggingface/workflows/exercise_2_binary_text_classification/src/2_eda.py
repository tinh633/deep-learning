"""Summarize label balance and pre-truncation lengths for train and validation."""

import argparse
from pathlib import Path

from config import VISUALIZATION
from utils import add_data_argument, data_metadata, write_json


def plot_eda(stats, lengths, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#007d7d", "#ca6a32"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    x = np.arange(2)
    for index, (name, color) in enumerate(zip(lengths, colors)):
        counts = stats["splits"][name]["label_counts"]
        bars = ax.bar(x + (index - 0.5) * 0.32, [counts["NEGATIVE"], counts["POSITIVE"]],
                      width=0.32, label=name.capitalize(), color=color)
        ax.bar_label(bars, padding=4)
    ax.set(xticks=x, xticklabels=["NEGATIVE", "POSITIVE"], ylabel="Reviews", title="Class distribution")
    ax.margins(y=0.18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(destination / "label_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    upper = max(stats["max_length"], *(max(values) for values in lengths.values()))
    bins = np.linspace(0, upper + 1, 41)
    for (name, values), color in zip(lengths.items(), colors):
        weights = np.full(len(values), 100 / len(values))
        ax.hist(values, bins=bins, weights=weights, histtype="step", linewidth=2,
                label=name.capitalize(), color=color)
    max_length = stats["max_length"]
    ax.axvline(max_length, color="#394150", linestyle="--", label=f"Limit · {max_length} tokens")
    ax.set(xlabel="Tokens per review", ylabel="Reviews in bin (%)", title="Review length before truncation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "token_lengths.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_argument(parser)
    args = parser.parse_args()
    meta = data_metadata(args.data_dir)

    import numpy as np
    from datasets import load_from_disk

    stats = {"snapshot": str(args.data_dir.resolve()), "max_length": meta["max_length"], "splits": {}}
    lengths = {}
    for name in ("train", "validation"):
        split = load_from_disk(str(args.data_dir / name))
        values = np.asarray(split["original_token_count"])
        lengths[name] = values
        stats["splits"][name] = {
            "count": len(split), "label_counts": meta["splits"][name]["label_counts"],
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": int(values.max()),
            "truncated_fraction": float((values > meta["max_length"]).mean()),
        }
    snapshot_id = meta["splits"]["train"]["sha256"][:12]
    figure_dir = f"eda/{snapshot_id}-tokens-{meta['max_length']}"
    stats["figure_dir"] = figure_dir
    destination = VISUALIZATION / figure_dir
    destination.mkdir(parents=True, exist_ok=True)
    plot_eda(stats, lengths, destination)
    summary = Path(args.data_dir) / "eda_summary.json"
    write_json(summary, stats)
    print(f"Summary: {summary}\nFigures: {destination}")


if __name__ == "__main__":
    main()
