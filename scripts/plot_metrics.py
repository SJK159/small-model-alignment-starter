"""Generate training plots from metrics.jsonl files for the report.

Usage:
    python scripts/plot_metrics.py --output_dir outputs/
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_metrics(path):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    reward_acc_steps, reward_accs = [], []
    reward_margin_steps, reward_margins = [], []

    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            step = entry["step"]
            if "train_loss" in entry:
                train_steps.append(step)
                train_losses.append(entry["train_loss"])
            if "val_loss" in entry:
                val_steps.append(step)
                val_losses.append(entry["val_loss"])
            if "reward_accuracy" in entry:
                reward_acc_steps.append(step)
                reward_accs.append(entry["reward_accuracy"])
            if "reward_margin" in entry:
                reward_margin_steps.append(step)
                reward_margins.append(entry["reward_margin"])

    return {
        "train": (train_steps, train_losses),
        "val": (val_steps, val_losses),
        "reward_acc": (reward_acc_steps, reward_accs),
        "reward_margin": (reward_margin_steps, reward_margins),
    }


def plot_sft(metrics_path, save_path):
    m = load_metrics(metrics_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(*m["train"], label="Train Loss", alpha=0.7)
    if m["val"][0]:
        ax.plot(*m["val"], label="Val Loss", marker="o", markersize=3)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_dpo(metrics_path, save_path, title="DPO Training"):
    m = load_metrics(metrics_path)
    n_plots = 1 + bool(m["reward_acc"][0]) + bool(m["reward_margin"][0])
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    axes[idx].plot(*m["train"], label="Train Loss", alpha=0.7)
    if m["val"][0]:
        axes[idx].plot(*m["val"], label="Val Loss", marker="o", markersize=3)
    axes[idx].set_xlabel("Step")
    axes[idx].set_ylabel("Loss")
    axes[idx].set_title(f"{title} — Loss")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

    if m["reward_acc"][0]:
        idx += 1
        axes[idx].plot(*m["reward_acc"], color="green", marker="o", markersize=3)
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel("Reward Accuracy")
        axes[idx].set_title(f"{title} — Reward Accuracy")
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, alpha=0.3)

    if m["reward_margin"][0]:
        idx += 1
        axes[idx].plot(*m["reward_margin"], color="orange", marker="o", markersize=3)
        axes[idx].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel("Reward Margin")
        axes[idx].set_title(f"{title} — Reward Margin")
        axes[idx].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--figures_dir", default="report/figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    sft_metrics = os.path.join(args.output_dir, "sft", "metrics.jsonl")
    if os.path.exists(sft_metrics):
        plot_sft(sft_metrics, os.path.join(args.figures_dir, "sft_loss.png"))

    for name, title in [("dpo_hh_rlhf", "HH-RLHF DPO"), ("dpo_ultrafeedback", "UltraFeedback DPO")]:
        metrics = os.path.join(args.output_dir, name, "metrics.jsonl")
        if os.path.exists(metrics):
            plot_dpo(metrics, os.path.join(args.figures_dir, f"{name}_metrics.png"), title)

    print("Done generating plots.")


if __name__ == "__main__":
    main()
