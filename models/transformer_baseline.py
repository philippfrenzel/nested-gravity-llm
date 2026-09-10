from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_sequence_length: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 128,
        **_: Dict,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.latest_metrics: Dict[str, float] = {}
        self.latest_trace: Dict[str, list] = {}

    def forward(self, tokens: torch.Tensor, return_metrics: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, sequence_length = tokens.shape
        positions = torch.arange(sequence_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.full((sequence_length, sequence_length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=causal_mask)
        logits = self.output_projection(encoded)
        self.latest_metrics = {
            "mean_local_force_norm": 0.0,
            "mean_nesting_force_norm": 0.0,
            "center_entropy": 0.0,
            "effective_num_centers": 0.0,
            "clipped_force_fraction": 0.0,
            "mean_token_center_distance": 0.0,
            "mean_token_token_distance": 0.0,
        }
        self.latest_trace = {}
        if return_metrics:
            return logits, dict(self.latest_metrics)
        return logits
