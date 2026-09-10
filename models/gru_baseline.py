from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        **_: Dict,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.latest_metrics: Dict[str, float] = {}

    def forward(self, tokens: torch.Tensor, return_metrics: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        embedded = self.embedding(tokens)
        hidden_states, _ = self.gru(embedded)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        self.latest_metrics = {
            "mean_local_force_norm": 0.0,
            "mean_nesting_force_norm": 0.0,
            "center_entropy": 0.0,
            "effective_num_centers": 0.0,
            "clipped_force_fraction": 0.0,
            "mean_token_center_distance": 0.0,
            "mean_token_token_distance": 0.0,
        }
        if return_metrics:
            return logits, dict(self.latest_metrics)
        return logits
