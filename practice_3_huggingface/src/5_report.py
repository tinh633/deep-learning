"""Create a reproducible Practice 3 report from saved run artifacts.

Run with: python practice_3_huggingface/src/5_report.py
No training, network calls, model loading, or fabricated results.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import EVAL_RUN, RUNS, ROOT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def fmt(value, percent=False, digits=2):
    if value is None:
        return "Not available"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:.4f}"


def history_per_epoch(history):
    """Trainer history: separate train/eval records; do not connect final recheck to epoch 3."""
    train, valid = {}, {}
    for row in history:
        epoch = row.get("epoch")
        if epoch is None or int(float(epoch)) != float(epoch):
            continue
        epoch = int(epoch)
        if "loss" in row and "eval_loss" not in row:
            train[epoch] = row["loss"]
        if "eval_loss" in row and "eval_f1" in row and epoch not in valid:
            # First logged evaluation is the epoch checkpoint; a later
            # post-training recheck may evaluate the restored best model.
            valid[epoch] = row
    return train, valid


def plot_loss(out, run, train, valid):
    fig, ax = plt.subplots(figsize=(9, 5.3))
    if train:
        ax.plot(sorted(train), [train[e] for e in sorted(train)], "o-", label="Training loss")
    if valid:
        ax.plot(sorted(valid), [valid[e]["eval_loss"] for e in sorted(valid)], "s-", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"DistilBERT · {run} | training vs validation")
    ax.set_xticks(sorted(set(train) | set(valid)))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=180)
    plt.close(fig)


def plot_validation(out, run, valid):
    if not valid:
        return
    epochs = sorted(valid)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.plot(epochs, [valid[e]["eval_accuracy"] * 100 for e in epochs], "o-", label="Validation accuracy")
    ax.plot(epochs, [valid[e]["eval_f1"] * 100 for e in epochs], "s-", label="Validation F1 (positive class)")
    ax.set(xlabel="Epoch", ylabel="Score (%)", title=f"DistilBERT · {run} | validation performance")
    ax.set_xticks(epochs)
    ax.set_ylim(75, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_scores.png", dpi=180)
    plt.close(fig)


def plot_confusion(out, run, evidence):
    cm = np.asarray(evidence["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["NEGATIVE", "POSITIVE"])
    ax.set_yticks([0, 1], ["NEGATIVE", "POSITIVE"])
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"{run} | final held-out test (n={cm.sum()})")
    cutoff = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cutoff else "black", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    run_dir = RUNS / EVAL_RUN
    history = read_json(run_dir / "history.json")
    config = read_json(run_dir / "config.json")
    final_val = read_json(run_dir / "validation_metrics.json")
    test = read_json(run_dir / "test_metrics.json")
    if history is None or config is None:
        raise SystemExit(f"Missing history/config for {run_dir}")
    if test is not None and test.get("split") != "test":
        raise SystemExit("test_metrics.json does not identify test split")
    out = ROOT / "visualization" / EVAL_RUN
    out.mkdir(parents=True, exist_ok=True)
    tr, va = history_per_epoch(history)
    plot_loss(out, EVAL_RUN, tr, va)
    plot_validation(out, EVAL_RUN, va)
    if test and "confusion_matrix" in test:
        plot_confusion(out, EVAL_RUN, test)

    best_epoch = max(va, key=lambda e: va[e]["eval_f1"]) if va else None
    min_loss_epoch = min(va, key=lambda e: va[e]["eval_loss"]) if va else None
    lines = [
        f"# Practice 3 — Experiment report: `{EVAL_RUN}`", "",
        "> Evidence-based report generated from saved training and evaluation artifacts. "
        "Training metrics, validation metrics and final test metrics are explicitly separated.", "",
        "## 1. Objective and method", "",
        "**Exercise 1:** use an off-the-shelf pretrained sentiment classifier for tokenization and inference; "
        "no training or test benchmark is claimed for this exercise.", "",
        "**Exercise 2:** fine-tune pretrained DistilBERT on labeled IMDb reviews for binary sentiment classification "
        "(NEGATIVE = 0, POSITIVE = 1). The validation set selects the checkpoint; the held-out test set "
        "is used for one final evaluation.", "",
        "## 2. Experiment configuration", "",
        f"- Pretrained backbone: `{config.get('base_model', 'not recorded')}`",
        f"- Dataset: `{config.get('dataset_id', 'not recorded')}`",
        f"- Samples: train **{config.get('train_samples','?')}**; validation **{config.get('validation_samples','?')}**; "
        f"test **{test.get('sample_count') if test else 'not yet evaluated'}**",
        f"- Epochs: **{config.get('epochs','?')}**; batch size: **{config.get('batch_size','?')}**; "
        f"learning rate: **{config.get('learning_rate','?')}**; weight decay: **{config.get('weight_decay','?')}**",
        f"- Max tokens per review: **{config.get('max_length','?')}**; random seed: **{config.get('seed','?')}**",
        f"- Checkpoint selection: **highest validation F1 for positive class**, per run configuration.", "",
        "## 3. Training and validation results", "",
        "| Epoch | Training loss* | Validation loss | Validation accuracy | Validation F1 (positive) |", 
        "|---:|---:|---:|---:|---:|",
    ]
    for e in sorted(set(tr) | set(va)):
        v = va.get(e, {})
        lines.append(f"| {e} | {fmt(tr.get(e))} | {fmt(v.get('eval_loss'))} | "
                     f"{fmt(v.get('eval_accuracy'), True)} | {fmt(v.get('eval_f1'), True)} |")
    lines += [
        "", "*Training losses above are epoch-level values logged by Hugging Face Trainer; "
        "they are not validation measurements.*", "",
        "![Training and validation loss](loss.png)", "",
        "![Validation accuracy and positive-class F1](validation_scores.png)", "",
    ]
    if best_epoch is not None:
        lines += [
            f"The highest validation F1 occurred at **epoch {best_epoch}** "
            f"(F1 **{fmt(va[best_epoch]['eval_f1'], True)}**, "
            f"accuracy **{fmt(va[best_epoch]['eval_accuracy'], True)}**). "
            f"The lowest validation loss occurred at **epoch {min_loss_epoch}** "
            f"({fmt(va[min_loss_epoch]['eval_loss'])}); selecting by F1 is different from selecting by loss.", "",
        ]
    if final_val:
        lines += [
            "The post-training validation recheck of the selected model recorded "
            f"accuracy **{fmt(final_val.get('eval_accuracy'), True)}**, "
            f"F1 **{fmt(final_val.get('eval_f1'), True)}**, and "
            f"loss **{fmt(final_val.get('eval_loss'))}**. "
            "This is still **validation**, not test performance.", "",
        ]
    lines += ["## 4. Final test results (held-out)", ""]
    if test:
        m = test["metrics"]
        report = test.get("classification_report", {})
        lines += [
            f"The frozen model was evaluated on **{test['sample_count']}** held-out IMDb reviews.", "",
            "| Metric | Final test |", "|---|---:|",
            f"| Accuracy | **{fmt(m.get('test_accuracy'), True)}** |",
            f"| Cross-entropy loss | {fmt(m.get('test_loss'))} |",
            f"| Precision (positive) | {fmt(m.get('test_precision'), True)} |",
            f"| Recall (positive) | {fmt(m.get('test_recall'), True)} |",
            f"| F1 (positive) | {fmt(m.get('test_f1'), True)} |",
            f"| Macro F1 | {fmt(report.get('macro avg', {}).get('f1-score'), True)} |",
            f"| ROC-AUC | {fmt(test.get('roc_auc'))} |", "",
            "### Per-class scores", "",
            "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
        ]
        for name in ("NEGATIVE", "POSITIVE"):
            r = report.get(name, {})
            lines.append(f"| {name} | {fmt(r.get('precision'), True)} | {fmt(r.get('recall'), True)} | "
                         f"{fmt(r.get('f1-score'), True)} | {r.get('support', 'N/A'):g} |")
        cm = np.asarray(test["confusion_matrix"])
        lines += ["", "### Confusion matrix (rows = true labels; columns = predictions)", "",
                  "| True \\ Predicted | NEGATIVE | POSITIVE |", "|---|---:|---:|",
                  f"| NEGATIVE | {cm[0,0]} | {cm[0,1]} |",
                  f"| POSITIVE | {cm[1,0]} | {cm[1,1]} |", "",
                  "![Confusion matrix on test data](confusion_matrix.png)", "",
                  f"Correct: **{int(np.trace(cm))}/{int(cm.sum())}**; "
                  f"false positives: **{int(cm[0,1])}**; false negatives: **{int(cm[1,0])}**.", ""]
    else:
        lines += ["No final test evidence is saved for this run. Final test metrics cannot be claimed.", ""]

    lines += ["## 5. Discussion", ""]
    if len(va) >= 3 and len(tr) >= 3:
        last = max(va)
        early = min(va)
        if va[last]["eval_loss"] > va[early]["eval_loss"] and tr[last] < tr[early]:
            lines.append(
                f"**Generalization gap:** epoch-level training loss fell from {fmt(tr[early])} "
                f"to {fmt(tr[last])}, while validation loss rose from "
                f"{fmt(va[early]['eval_loss'])} to {fmt(va[last]['eval_loss'])}. "
                "This is evidence consistent with overfitting; it does not establish the cause by itself."
            )
            lines.append("")
    if test:
        m=test["metrics"]
        if final_val and "eval_accuracy" in final_val:
            diff = 100 * (final_val["eval_accuracy"] - m["test_accuracy"])
            lines.append(f"**Generalization to test data:** final test accuracy "
                         f"({fmt(m['test_accuracy'], True)}) was **{diff:.2f} percentage points** "
                         f"below selected-checkpoint validation accuracy ({fmt(final_val['eval_accuracy'], True)}). "
                         "Sampling variability and differences between subsets may contribute; no causal explanation is proven.")
            lines.append("")
        cm=np.asarray(test["confusion_matrix"])
        lines.append(f"**Asymmetric errors:** {cm[0,1]} NEGATIVE reviews were incorrectly labeled POSITIVE, "
                     f"versus {cm[1,0]} POSITIVE reviews incorrectly labeled NEGATIVE. "
                     "The classifier detects positive reviews more reliably (by recall) than negative reviews.")
        lines.append("")
    lines += [
        "## 6. Limitations and validity", "",
        "- **Limited data:** the run used only a small subset of IMDb; the test set contains 500 samples, "
        "so estimated performance is subject to sampling variation.",
        "- **Single seed:** only this run is analyzed; replicate seeds would help quantify stability.",
        "- **Truncation:** input is limited to the configured number of tokens; longer reviews may lose sentiment cues.",
        "- **Dataset specificity:** IMDb reviews do not establish transfer to social media, other languages, or other domains.",
        "- **Model selection:** best checkpoint was selected on validation F1, not validation loss; "
        "do not tune hyperparameters based on this final test set.",
        "- **Calibration:** high classification confidence is not proof of well-calibrated probabilities.", "",
        "## 7. Conclusion", "",
        "Both exercises demonstrate Hugging Face usage: pretrained inference (Exercise 1) and "
        "supervised fine-tuning plus held-out evaluation (Exercise 2). "
        "The reported figures are observational results for this run, not guarantees of general performance.", "",
        "## 8. Traceable artifacts", "",
        "- `outputs/runs/<run>/config.json` — saved hyperparameters and selection rule.",
        "- `outputs/runs/<run>/history.json` and `train.log` — recorded Trainer history.",
        "- `outputs/runs/<run>/validation_metrics.json` — selected-checkpoint validation recheck.",
        "- `outputs/runs/<run>/test_metrics.json` — final test metrics and per-example predictions.",
        "- `outputs/runs/<run>/model/` — selected fine-tuned model and tokenizer.",
        "- `visualization/<run>/` — this report and generated figures.", "",
    ]
    report_path = out / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Report:", report_path)
    print("Training epochs:", len(va))
    print("Final test:", "available" if test else "not available")


if __name__ == "__main__":
    main()
