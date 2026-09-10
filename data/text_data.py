from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .tokenizer import CharacterTokenizer


DEFAULT_TEXT = "\n".join(
    [
        "Der Mond steht über dem stillen See.",
        "Die Stadt erwacht, während der Wind durch die Straßen zieht.",
        "Zwischen alten Häusern klingt leise ein entferntes Lied.",
        "Am Morgen glitzern Tropfen auf dem Gras wie kleine Sterne.",
    ]
    * 128
)


@dataclass
class CharacterTextDataset:
    text: str
    sequence_length: int = 128
    seed: int = 42

    @classmethod
    def from_path(cls, path: str | None, sequence_length: int = 128, seed: int = 42) -> "CharacterTextDataset":
        source = Path(path) if path else Path("data/text.txt")
        if source.exists():
            text = source.read_text(encoding="utf-8")
        else:
            text = DEFAULT_TEXT
        return cls(text=text, sequence_length=sequence_length, seed=seed)

    @property
    def tokenizer(self) -> CharacterTokenizer:
        return CharacterTokenizer.from_text(self.text)

    def split(self) -> Dict[str, torch.Tensor]:
        encoded = np.array(self.tokenizer.encode(self.text), dtype=np.int64)
        n = len(encoded)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        return {
            "train": torch.tensor(encoded[:train_end], dtype=torch.long),
            "val": torch.tensor(encoded[train_end:val_end], dtype=torch.long),
            "test": torch.tensor(encoded[val_end:], dtype=torch.long),
        }

    def sample_batch(self, split: str, batch_size: int) -> Dict[str, torch.Tensor]:
        data = self.split()[split]
        generator = np.random.default_rng(self.seed + {"train": 0, "val": 1, "test": 2}[split])
        starts = generator.integers(0, len(data) - self.sequence_length - 1, size=batch_size)
        sequences = [data[start : start + self.sequence_length + 1] for start in starts]
        stacked = torch.stack(sequences, dim=0)
        targets = stacked[:, 1:].clone()
        target_mask = torch.ones_like(targets, dtype=torch.bool)
        return {"inputs": stacked[:, :-1], "targets": targets, "target_mask": target_mask}
