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


def causality_profile(model, tokens: torch.Tensor) -> np.ndarray:
    if tokens.size(1) < 3:
        return np.zeros(tokens.size(1), dtype=float)
    modified = tokens.clone()
    modified[:, -1] = (modified[:, -1] + 1) % max(2, int(tokens.max().item()) + 2)
    with torch.no_grad():
        baseline = model(tokens)
        changed = model(modified)
    return (baseline - changed).abs().amax(dim=-1)[0].cpu().numpy()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = ExperimentConfig(**checkpoint["config"])
    config = ExperimentConfig(**checkpoint["config"])
    if args.task and args.task != checkpoint_config.task:
        raise ValueError(f"Checkpoint task is '{checkpoint_config.task}', cannot visualize as '{args.task}'")
    if args.task:
        config.task = args.task
    set_seed(config.seed)
    text_dataset = CharacterTextDataset.from_path(config.text_path, config.text_sequence_length, config.seed) if config.task == "text" else None
    if config.task == "text" and text_dataset is not None:
        config.vocab_size = len(text_dataset.tokenizer.vocab)
    model = make_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics_csv = Path(checkpoint_config.output_root) / "metrics" / f"{experiment_name(checkpoint_config)}.csv"
    plot_dir = Path(config.output_root) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_csv}")
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
    trace = getattr(model, "latest_trace", {})

    accuracy_candidates = [key for key in rows[-1].keys() if key.startswith("validation_accuracy_at_gap_")]
    recall_candidates = [key for key in rows[-1].keys() if key.startswith("validation_accuracy_num_pairs_")]
    if accuracy_candidates:
        plt.figure()
        pairs = sorted((int(key.rsplit("_", 1)[-1]), float(rows[-1][key])) for key in accuracy_candidates)
        gaps = [pair[0] for pair in pairs]
        values = [pair[1] for pair in pairs]
        plt.plot(gaps, values, marker="o")
        plt.xlabel("gap")
        plt.ylabel("accuracy")
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_by_gap.png")
        plt.close()
    elif recall_candidates:
        plt.figure()
        pairs = sorted((int(key.rsplit("_", 1)[-1]), float(rows[-1][key])) for key in recall_candidates)
        pair_counts = [pair[0] for pair in pairs]
        values = [pair[1] for pair in pairs]
        plt.plot(pair_counts, values, marker="o")
        plt.xlabel("num_pairs")
        plt.ylabel("accuracy")
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_by_num_pairs.png")
        plt.close()
    elif {"validation_token_accuracy", "validation_full_sequence_accuracy"} <= set(rows[-1].keys()):
        plt.figure()
        labels = ["token_accuracy", "full_sequence_accuracy"]
        values = [float(rows[-1]["validation_token_accuracy"]), float(rows[-1]["validation_full_sequence_accuracy"])]
        plt.bar(labels, values)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_summary.png")
        plt.close()

    if hasattr(model, "centers"):
        centers = model.centers.detach().cpu().numpy()
        token_positions = np.array(trace.get("positions", []), dtype=float)
        if len(token_positions) > 0:
            combined = np.concatenate([token_positions, centers], axis=0)
            projected = pca_2d(combined)
            projected_tokens = projected[: len(token_positions)]
            projected_centers = projected[len(token_positions) :]
            plt.figure()
            plt.scatter(projected_tokens[:, 0], projected_tokens[:, 1], c=np.arange(len(projected_tokens)), s=20, label="tokens")
            plt.scatter(projected_centers[:, 0], projected_centers[:, 1], c="red", marker="x", s=80, label="centers")
            plt.legend()
            plt.title("gravity centers and token positions")
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

        entropy_by_position = np.array(trace.get("center_entropy_by_position", []), dtype=float)
        if len(entropy_by_position) > 0:
            plt.figure()
            plt.plot(entropy_by_position)
            plt.xlabel("position")
            plt.ylabel("center entropy")
            plt.tight_layout()
            plt.savefig(plot_dir / "center_entropy.png")
            plt.close()

        local_force_norms = np.array(trace.get("local_force_norms_by_position", []), dtype=float)
        nesting_force_norms = np.array(trace.get("nesting_force_norms_by_position", []), dtype=float)
        if len(local_force_norms) > 0 and len(nesting_force_norms) > 0:
            plt.figure()
            plt.plot(local_force_norms, label="local")
            plt.plot(nesting_force_norms, label="nesting")
            plt.legend()
            plt.xlabel("position")
            plt.ylabel("force norm")
            plt.tight_layout()
            plt.savefig(plot_dir / "force_norms.png")
            plt.close()

        interaction = np.array(trace.get("interaction_strength", []), dtype=float)
        if interaction.size > 0:
            plt.figure()
            plt.imshow(interaction, aspect="auto", origin="lower")
            plt.colorbar(label="gravitational interaction strength")
            plt.tight_layout()
            plt.savefig(plot_dir / "gravitational_interaction_strength.png")
            plt.close()

    causality = causality_profile(model, batch["inputs"][:1])
    plt.figure()
    plt.plot(np.arange(len(causality)), causality)
    plt.xlabel("position")
    plt.ylabel("max abs logit diff")
    plt.tight_layout()
    plt.savefig(plot_dir / "causality_check.png")
    plt.close()


if __name__ == "__main__":
    main()
