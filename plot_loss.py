#!/usr/bin/env python3
"""Plot loss curves from training CSV files.

Usage:
    python3 plot_loss.py                    # auto-discover all output/*/loss.csv
    python3 plot_loss.py --out output/loss  # custom output prefix
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    steps, losses = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def smooth(values, weight=0.95):
    out = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        out.append(last)
    return out


def label_from_path(path):
    name = os.path.basename(os.path.dirname(path))
    return name.replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/loss", help="output prefix (no extension)")
    parser.add_argument("--smooth", type=float, default=0.95, help="EMA smoothing weight")
    args = parser.parse_args()

    csvs = sorted(glob.glob("output/*/loss.csv"))
    if not csvs:
        print("No loss.csv files found in output/*/")
        return

    ddpm_csvs = [c for c in csvs if "ddpm" in c.lower()]
    flow_csvs = [c for c in csvs if "flow" in c.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    for ax, group, title in [
        (axes[0], ddpm_csvs, "DDPM"),
        (axes[1], flow_csvs, "Flow Matching"),
    ]:
        for i, path in enumerate(group):
            steps, losses = read_csv(path)
            if not steps:
                continue
            smoothed = smooth(losses, args.smooth)
            label = label_from_path(path)
            c = colors[i % len(colors)]
            ax.plot(steps, losses, alpha=0.15, color=c, linewidth=0.5)
            ax.plot(steps, smoothed, label=label, color=c, linewidth=1.8)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(args.out + ".svg", bbox_inches="tight")
    print(f"Wrote {args.out}.png and {args.out}.svg")


if __name__ == "__main__":
    main()
