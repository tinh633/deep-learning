"""Create training curves and evaluation figures for one completed run."""

import argparse

from config import VISUALIZATION
from utils import completed_run, read_json


def plot_history(history, best_step, destination):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    train = [row for row in history if "loss" in row and "epoch" in row]
    validation_by_step = {}
    for row in history:
        if "eval_loss" in row and "step" in row:
            # Trainer's final evaluate() may log again at its last step.
            validation_by_step.setdefault(row["step"], row)
    validation = [validation_by_step[step] for step in sorted(validation_by_step)]
    if not train or not validation:
        raise SystemExit("No epoch loss history found.")

    fig, (loss, metrics) = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    epochs = [row["epoch"] for row in validation]
    loss.plot([row["epoch"] for row in train], [row["loss"] for row in train],
              "o-", color="#ca6a32", label="Train")
    loss.plot(epochs, [row["eval_loss"] for row in validation], "o-", color="#007d7d", label="Validation")
    metrics.plot(epochs, [row["eval_accuracy"] for row in validation], "o-", color="#3c69a8", label="Accuracy")
    metrics.plot(epochs, [row["eval_f1"] for row in validation], "o-", color="#007d7d", label="F1 · POSITIVE")
    best = validation_by_step.get(best_step)
    for ax in (loss, metrics):
        if best is not None:
            ax.axvline(best["epoch"], color="#394150", linestyle="--", alpha=.6, label="Selected checkpoint")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    loss.set(title="Cross-entropy loss", ylabel="Loss")
    metrics.set(title="Validation scores", ylim=(0, 1.03))
    metrics.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(destination / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_evaluation(evidence, destination):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    split = evidence["split"]
    matrix = np.asarray(evidence["confusion_matrix"])
    labels = evidence["label_order"]
    total = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, total, out=np.zeros_like(matrix, dtype=float), where=total != 0)
    fig, (ax, scores) = plt.subplots(1, 2, figsize=(10.6, 4.1), layout="constrained",
                                     gridspec_kw={"width_ratios": [1, 1.4]})
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row, col in np.ndindex(matrix.shape):
        ax.text(col, row, f"{matrix[row, col]}\n{normalized[row, col]:.1%}",
                ha="center", va="center", color="white" if normalized[row, col] > .55 else "#14213d")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="Actual label", title="Count and share of actual class")
    names = ["accuracy", "precision", "recall", "f1"]
    values = [evidence["metrics"][f"{split}_{name}"] for name in names]
    bars = scores.barh(["Accuracy", "Precision", "Recall", "F1"], values, height=.44, color="#007d7d")
    scores.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=5)
    scores.invert_yaxis()
    scores.set(xlim=(0, 1.17), xticks=[0, .25, .5, .75, 1], title="Evaluation metrics")
    scores.xaxis.set_major_formatter(PercentFormatter(1))
    scores.grid(axis="x", alpha=.2)
    scores.set_axisbelow(True)
    fig.suptitle(f"{evidence['run']} · {split.capitalize()} · n={evidence['sample_count']}", weight="bold")
    fig.savefig(destination / f"{split}_evaluation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run = completed_run(args.run)
    evidence_path = run / f"{args.split}_evaluation.json"
    if not evidence_path.is_file():
        raise SystemExit(f"Run src/4_evaluate.py --run {args.run} --split {args.split} first.")

    import matplotlib
    matplotlib.use("Agg")

    destination = VISUALIZATION / args.run
    destination.mkdir(parents=True, exist_ok=True)
    plot_history(read_json(run / "history.json"), read_json(run / "config.json")["best_step"], destination)
    plot_evaluation(read_json(evidence_path), destination)
    print(f"Figures saved in {destination}")


if __name__ == "__main__":
    main()
