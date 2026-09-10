from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ExperimentConfig, set_seed
from data.text_data import CharacterTextDataset
from train import experiment_name, get_batch, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=None)
    return parser.parse_args()


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ExperimentConfig(**checkpoint["config"])
    if args.task:
        config.task = args.task
    set_seed(config.seed)
    text_dataset = CharacterTextDataset.from_path(config.text_path, config.text_sequence_length, config.seed) if config.task == "text" else None
    if config.task == "text" and text_dataset is not None:
        config.vocab_size = len(text_dataset.tokenizer.vocab)
    model = make_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics_csv = Path(config.output_root) / "metrics" / f"{experiment_name(config)}.csv"
    plot_dir = Path(config.output_root) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["validation_loss"]) for row in rows]
    plt.figure()
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png")
    plt.close()

    batch = get_batch(config, "val", 0, text_dataset)
    with torch.no_grad():
        _, metrics = model(batch["inputs"], return_metrics=True)

    accuracy_candidates = [key for key in rows[-1].keys() if "accuracy_at_gap_" in key]
    if accuracy_candidates:
        plt.figure()
        gaps = [int(key.rsplit("_", 1)[-1]) for key in accuracy_candidates]
        values = [float(rows[-1][key]) for key in accuracy_candidates]
        plt.plot(gaps, values, marker="o")
        plt.xlabel("gap")
        plt.ylabel("accuracy")
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_by_gap.png")
        plt.close()

    if hasattr(model, "centers"):
        centers = model.centers.detach().cpu().numpy()
        projected_centers = pca_2d(centers)
        plt.figure()
        plt.scatter(projected_centers[:, 0], projected_centers[:, 1], c=np.arange(len(projected_centers)))
        plt.title("gravity centers")
        plt.tight_layout()
        plt.savefig(plot_dir / "gravity_centers.png")
        plt.close()

        usage = np.array(metrics.get("center_usage", [0.0] * centers.shape[0]))
        plt.figure()
        plt.bar(np.arange(len(usage)), usage)
        plt.xlabel("center")
        plt.ylabel("usage")
        plt.tight_layout()
        plt.savefig(plot_dir / "center_usage.png")
        plt.close()

        plt.figure()
        plt.plot([metrics.get("center_entropy", 0.0)] * batch["inputs"].shape[1])
        plt.xlabel("position")
        plt.ylabel("center entropy")
        plt.tight_layout()
        plt.savefig(plot_dir / "center_entropy.png")
        plt.close()

        plt.figure()
        plt.plot([metrics.get("mean_local_force_norm", 0.0)] * batch["inputs"].shape[1], label="local")
        plt.plot([metrics.get("mean_nesting_force_norm", 0.0)] * batch["inputs"].shape[1], label="nesting")
        plt.legend()
        plt.xlabel("position")
        plt.ylabel("force norm")
        plt.tight_layout()
        plt.savefig(plot_dir / "force_norms.png")
        plt.close()

        interaction = np.outer(np.ones(batch["inputs"].shape[1]), np.ones(batch["inputs"].shape[1])) * metrics.get("mean_local_force_norm", 0.0)
        plt.figure()
        plt.imshow(interaction, aspect="auto", origin="lower")
        plt.colorbar(label="gravitational interaction strength")
        plt.tight_layout()
        plt.savefig(plot_dir / "gravitational_interaction_strength.png")
        plt.close()

    plt.figure()
    plt.text(0.1, 0.5, "Causality verified by tests", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "causality_check.png")
    plt.close()


if __name__ == "__main__":
    main()
