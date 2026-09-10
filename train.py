from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from config import build_config, save_json, set_seed
from data.synthetic import generate_associative_recall_batch, generate_brackets_batch, generate_copy_batch
from data.text_data import CharacterTextDataset
from models import CausalTransformerLM, GRULanguageModel, NestedGravitationalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="nested_gravity")
    parser.add_argument("--task", type=str, default="copy", choices=["copy", "associative_recall", "brackets", "text"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--gap-length", type=int, default=None)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--text-path", type=str, default=None)
    return parser.parse_args()


def make_model(config) -> nn.Module:
    kwargs = config.to_dict()
    if config.model in {"nested_gravity", "local_gravity", "nesting_only", "gru_only", "nested_gravity_repulsion"}:
        return NestedGravitationalLM(**kwargs)
    if config.model == "gru":
        return GRULanguageModel(**kwargs)
    if config.model == "transformer":
        return CausalTransformerLM(**kwargs)
    raise ValueError(f"Unknown model: {config.model}")


def experiment_name(config) -> str:
    suffix = {
        "copy": f"gap{config.gap_length}",
        "associative_recall": f"pairs{config.num_pairs}",
        "brackets": f"depth{config.max_depth}",
        "text": "char",
    }[config.task]
    return f"{config.model}_{config.task}_{suffix}"


def get_batch(config, split: str, step: int, text_dataset: CharacterTextDataset | None = None) -> Dict[str, torch.Tensor]:
    seed_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    seed = config.seed + step + seed_offsets.get(split, 10_000)
    if config.task == "copy":
        return generate_copy_batch(config.batch_size, config.memory_length, config.gap_length, config.vocab_size, seed)
    if config.task == "associative_recall":
        return generate_associative_recall_batch(config.batch_size, config.num_pairs, max(config.vocab_size, 64), seed)
    if config.task == "brackets":
        return generate_brackets_batch(config.batch_size, config.max_depth, config.noise_tokens, seed)
    if config.task == "text":
        assert text_dataset is not None
        return text_dataset.sample_batch(split, config.batch_size, step=step)
    raise ValueError(config.task)


def sequence_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    targets_are_aligned: bool = False,
) -> Dict[str, float]:
    aligned_logits, aligned_targets, aligned_mask = align_for_loss(logits, targets, target_mask, targets_are_aligned)
    vocab_size = logits.size(-1)
    masked_logits = aligned_logits[aligned_mask]
    masked_targets = aligned_targets[aligned_mask]
    if masked_targets.numel() == 0:
        loss = aligned_logits.sum() * 0.0
    else:
        loss = F.cross_entropy(masked_logits.reshape(-1, vocab_size), masked_targets.reshape(-1))
    predictions = aligned_logits.argmax(dim=-1)
    correct = (predictions == aligned_targets) & aligned_mask
    accuracy = correct.float().sum() / aligned_mask.float().sum().clamp_min(1.0)
    metrics = {
        "loss": loss,
        "accuracy": accuracy.item(),
        "perplexity": math.exp(min(20.0, loss.item())),
    }
    return metrics


def align_for_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    targets_are_aligned: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if targets_are_aligned:
        return logits, targets, target_mask
    return logits[:, :-1], targets[:, 1:], target_mask[:, 1:]


def extra_task_metrics(
    config,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    targets_are_aligned: bool = False,
    sequence_loss: float | None = None,
) -> Dict[str, float]:
    aligned_logits, aligned_targets, aligned_mask = align_for_loss(logits, targets, target_mask, targets_are_aligned)
    predictions = aligned_logits.argmax(dim=-1)
    selected_correct = ((predictions == aligned_targets) & aligned_mask).float()
    selected_total = aligned_mask.float().sum().clamp_min(1.0)
    metrics: Dict[str, float] = {}
    if config.task == "copy":
        metrics[f"accuracy_at_gap_{config.gap_length}"] = (selected_correct.sum() / selected_total).item()
    elif config.task == "associative_recall":
        metrics[f"accuracy_num_pairs_{config.num_pairs}"] = (selected_correct.sum() / selected_total).item()
    elif config.task == "brackets":
        token_accuracy = (selected_correct.sum() / selected_total).item()
        full_sequence = (((predictions == aligned_targets) | ~aligned_mask).all(dim=1)).float().mean().item()
        errors_per_sequence = ((predictions != aligned_targets) & aligned_mask).float().sum(dim=1).mean().item()
        metrics.update(
            {
                "token_accuracy": token_accuracy,
                "full_sequence_accuracy": full_sequence,
                "bracket_errors_per_sequence": errors_per_sequence,
            }
        )
    elif config.task == "text":
        loss_value = sequence_loss if sequence_loss is not None else sequence_metrics(
            logits,
            targets,
            target_mask,
            targets_are_aligned,
        )["loss"].item()
        metrics["bits_per_character"] = loss_value / math.log(2)
    return metrics


def regularization_loss(config, model: nn.Module, metrics: Dict[str, float]) -> torch.Tensor:
    if not isinstance(model, NestedGravitationalLM):
        return next(model.parameters()).new_tensor(0.0)
    entropy = torch.tensor(metrics.get("center_entropy", 0.0), device=model.centers.device)
    return (
        config.center_loss_weight * model.center_norm_regularization()
        + config.repulsion_loss_weight * model.repulsion_regularization()
        + config.entropy_loss_weight * entropy
    )


def train_or_eval_epoch(config, model: nn.Module, optimizer: AdamW | None, split: str, text_dataset: CharacterTextDataset | None = None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    num_steps = config.steps_per_epoch if is_train else config.val_steps
    totals: Dict[str, float] = {
        "loss": 0.0,
        "accuracy": 0.0,
        "gradient_norm": 0.0,
        "mean_local_force_norm": 0.0,
        "mean_nesting_force_norm": 0.0,
        "center_entropy": 0.0,
        "effective_num_centers": 0.0,
        "clipped_force_fraction": 0.0,
        "mean_token_center_distance": 0.0,
        "mean_token_token_distance": 0.0,
    }
    extra_totals: Dict[str, float] = {}
    for step in range(num_steps):
        batch = get_batch(config, split, step, text_dataset)
        inputs = batch["inputs"]
        targets = batch["targets"]
        target_mask = batch["target_mask"]
        targets_are_aligned = batch.get("targets_are_aligned", False)
        with torch.set_grad_enabled(is_train):
            logits, model_metrics = model(inputs, return_metrics=True)
            metrics = sequence_metrics(logits, targets, target_mask, targets_are_aligned)
            loss = metrics["loss"] + regularization_loss(config, model, model_metrics)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                totals["gradient_norm"] += float(gradient_norm)
        totals["loss"] += float(metrics["loss"].item())
        totals["accuracy"] += float(metrics["accuracy"])
        for key in list(totals.keys())[3:]:
            totals[key] += float(model_metrics.get(key, 0.0))
        batch_extra = extra_task_metrics(
            config,
            logits,
            targets,
            target_mask,
            targets_are_aligned,
            sequence_loss=float(metrics["loss"].item()),
        )
        for key, value in batch_extra.items():
            extra_totals[key] = extra_totals.get(key, 0.0) + float(value)
    averaged = {key: value / num_steps for key, value in totals.items()}
    averaged.update({key: value / num_steps for key, value in extra_totals.items()})
    averaged["perplexity"] = math.exp(min(20.0, averaged["loss"]))
    if config.task == "text":
        averaged["bits_per_character"] = averaged["loss"] / math.log(2)
    return averaged


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)
    name = experiment_name(config)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)

    text_dataset = CharacterTextDataset.from_path(config.text_path, config.text_sequence_length, config.seed) if config.task == "text" else None
    if config.task == "text" and text_dataset is not None:
        config.vocab_size = len(text_dataset.tokenizer.vocab)
    model = make_model(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    metrics_path = output_root / "metrics" / f"{name}.csv"
    summary_path = output_root / "metrics" / f"{name}.json"
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else output_root / "checkpoints" / f"{name}_best.pt"

    history = []
    best_val_loss = float("inf")
    patience = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, config.epochs + 1):
            train_metrics = train_or_eval_epoch(config, model, optimizer, "train", text_dataset)
            val_metrics = train_or_eval_epoch(config, model, None, "val", text_dataset)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "gradient_norm": train_metrics["gradient_norm"],
                "mean_local_force_norm": train_metrics["mean_local_force_norm"],
                "mean_nesting_force_norm": train_metrics["mean_nesting_force_norm"],
                "center_entropy": train_metrics["center_entropy"],
            }
            for source, prefix in ((train_metrics, "train"), (val_metrics, "validation")):
                for key, value in source.items():
                    row.setdefault(f"{prefix}_{key}", value)
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience = 0
                torch.save({"model_state": model.state_dict(), "config": config.to_dict(), "history": history[-5:]}, checkpoint_path)
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    payload = {
        "experiment": name,
        "best_validation_loss": best_val_loss,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "config": config.to_dict(),
    }
    save_json(summary_path, payload)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
