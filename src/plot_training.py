import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"  # blue
VAL_COLOR = "#eb6834"    # orange
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"

LOSSES = [("box_loss", "Box loss"), ("cls_loss", "Class loss"),
          ("dfl_loss", "DFL loss"), ("angle_loss", "Angle loss")]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Ultralytics training vs validation curves from results.csv")
    parser.add_argument("run_dir", type=str, help="Training run folder containing results.csv")
    parser.add_argument("--out", type=str, default=None, help="Output PNG (default: <run_dir>/results_combined.png)")
    return parser.parse_args()


def load(run_dir):
    with open(Path(run_dir) / "results.csv") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel("Epoch", color=INK_MUTED, fontsize=9)


def main():
    args = parse_args()
    d = load(args.run_dir)
    ep = d["epoch"]
    best = max(range(len(ep)), key=lambda i: d["metrics/mAP50(B)"][i])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    flat = axes.flatten()

    for ax, (key, title) in zip(flat, LOSSES):
        ax.plot(ep, d[f"train/{key}"], color=TRAIN_COLOR, lw=2, label="Train")
        ax.plot(ep, d[f"val/{key}"], color=VAL_COLOR, lw=2, label="Validation")
        style(ax, title)

    ax = flat[4]
    ax.plot(ep, d["metrics/mAP50(B)"], color=VAL_COLOR, lw=2, label="mAP50")
    ax.plot(ep, d["metrics/mAP50-95(B)"], color=VAL_COLOR, lw=2, ls="--", label="mAP50-95")
    ax.set_ylim(0, 1)
    style(ax, "Validation mAP")

    # Precision / recall (validation)
    ax = flat[5]
    ax.plot(ep, d["metrics/precision(B)"], color=VAL_COLOR, lw=2, label="Precision")
    ax.plot(ep, d["metrics/recall(B)"], color=VAL_COLOR, lw=2, ls=":", label="Recall")
    ax.set_ylim(0, 1)
    style(ax, "Validation precision / recall")

    for ax in flat:
        ax.axvline(ep[best], color=INK_MUTED, lw=1, ls=(0, (2, 3)))
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="best")

    fig.suptitle(f"{Path(args.run_dir).resolve().name}  |  best epoch {int(ep[best])} "
                 f"(mAP50 {d['metrics/mAP50(B)'][best]:.3f}, dotted vertical line)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out) if args.out else Path(args.run_dir) / "results_combined.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
