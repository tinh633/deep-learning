"""Plot both class scores from the last Exercise 1 prediction."""

import argparse
import json
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parents[1]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = ROOT / "outputs" / "sentiment_result.json"
    if not source.is_file():
        raise SystemExit("Run src/1_sentiment.py first to produce sentiment_result.json.")
    result = json.loads(source.read_text(encoding="utf-8"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = ["NEGATIVE", "POSITIVE"]
    values = [result["scores"][label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.2, 3.5), layout="constrained")
    bars = ax.barh(labels, values, height=0.42, color=["#ad4a58", "#007d7d"])
    ax.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=6)
    ax.set(xlim=(0, 1.16), xticks=[0, .25, .5, .75, 1], xlabel="Softmax score")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_title(shorten(result["sentence"], width=92, placeholder="…"), loc="left", fontsize=10)
    fig.suptitle(f"Sentiment prediction · {result['predicted_label']}", x=0.08, ha="left", weight="bold")
    path = ROOT / "visualization" / "sentiment_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
