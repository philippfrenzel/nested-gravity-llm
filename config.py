from __future__ import annotations

import ast
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    model: str = "nested_gravity"
    task: str = "copy"
    seed: int = 42
    vocab_size: int = 32
    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    max_sequence_length: int = 256
    gravity_dim: int = 16
    num_centers: int = 8
    local_window: int = 16
    epsilon: float = 0.1
    gravity_power: float = 1.0
    force_clip: float = 5.0
    temperature: float = 0.5
    ema_alpha: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    epochs: int = 20
    batch_size: int = 64
    steps_per_epoch: int = 100
    val_steps: int = 20
    early_stopping_patience: int = 5
    memory_length: int = 8
    gap_length: int = 32
    num_pairs: int = 4
    max_depth: int = 4
    noise_tokens: int = 0
    text_sequence_length: int = 128
    center_loss_weight: float = 1e-3
    entropy_loss_weight: float = 0.0
    repulsion_loss_weight: float = 0.0
    use_local_gravity: bool = True
    use_nesting: bool = True
    use_repulsion: bool = False
    output_root: str = "outputs"
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    text_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw.strip().strip('"').strip("'")


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a flat YAML-compatible `key: value` config without extra dependencies."""
    config: Dict[str, Any] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        if line[:1].isspace() or stripped.startswith("- "):
            raise ValueError(f"Unsupported nested/list config syntax at {path}:{line_number}")
        key, value = stripped.split(":", 1)
        if ":" in value and not value.strip().startswith(("'", "\"")):
            raise ValueError(f"Ambiguous config value at {path}:{line_number}; quote values containing ':'")
        config[key.strip()] = _coerce_value(value.strip())
    return config


def build_config(args: Any) -> ExperimentConfig:
    config = ExperimentConfig()
    config_path = getattr(args, "config", None)
    if config_path:
        config.config_path = config_path
        for key, value in load_config_file(config_path).items():
            if hasattr(config, key):
                setattr(config, key, value)
    for key in config.to_dict().keys():
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                setattr(config, key, value)
    if config.model == "local_gravity":
        config.use_local_gravity = True
        config.use_nesting = False
    elif config.model == "nesting_only":
        config.use_local_gravity = False
        config.use_nesting = True
    elif config.model == "gru_only":
        config.use_local_gravity = False
        config.use_nesting = False
    elif config.model == "nested_gravity_repulsion":
        config.use_local_gravity = True
        config.use_nesting = True
        config.use_repulsion = True
    return config


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
