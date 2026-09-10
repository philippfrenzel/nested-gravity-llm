from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ExperimentConfig, save_json, set_seed
from data.text_data import CharacterTextDataset
from train import experiment_name, get_batch, make_model, sequence_metrics, extra_task_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    config = ExperimentConfig(**payload["config"])
    if args.task is not None:
        config.task = args.task
    set_seed(config.seed)
    text_dataset = CharacterTextDataset.from_path(config.text_path, config.text_sequence_length, config.seed) if config.task == "text" else None
    if config.task == "text" and text_dataset is not None:
        config.vocab_size = len(text_dataset.tokenizer.vocab)
    model = make_model(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    batch = get_batch(config, args.split, 0, text_dataset)
    with torch.no_grad():
        logits, model_metrics = model(batch["inputs"], return_metrics=True)
    metrics = sequence_metrics(logits, batch["targets"], batch["target_mask"])
    metrics.update(extra_task_metrics(config, logits, batch["targets"], batch["target_mask"]))
    metrics.update(model_metrics)
    metrics["loss"] = float(metrics["loss"].item())
    output_path = Path(config.output_root) / "metrics" / f"{experiment_name(config)}_{args.split}_evaluation.json"
    save_json(output_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
