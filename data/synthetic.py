from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


COPY_DELIMITER = 1
COPY_GAP = 0
ASSOCIATIVE_QUERY = 1
ASSOCIATIVE_ANSWER = 2
BRACKET_PAD = 0
BRACKET_OPEN = {1: 2, 3: 4, 5: 6}
BRACKET_CLOSE = {2: 1, 4: 3, 6: 5}


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def generate_copy_batch(
    batch_size: int,
    memory_length: int = 8,
    gap_length: int = 32,
    vocab_size: int = 32,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    generator = _generator(seed)
    memory = torch.randint(2, vocab_size, (batch_size, memory_length), generator=generator)
    gap = torch.full((batch_size, gap_length), COPY_GAP, dtype=torch.long)
    delimiter = torch.full((batch_size, 1), COPY_DELIMITER, dtype=torch.long)
    tokens = torch.cat([memory, delimiter, gap, delimiter, memory], dim=1)
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    target_mask[:, -memory_length:] = True
    return {"inputs": tokens, "targets": tokens.clone(), "target_mask": target_mask}


def generate_associative_recall_batch(
    batch_size: int,
    num_pairs: int = 4,
    vocab_size: int = 64,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    generator = _generator(seed)
    keys = torch.randint(3, vocab_size // 2, (batch_size, num_pairs), generator=generator)
    values = torch.randint(vocab_size // 2, vocab_size, (batch_size, num_pairs), generator=generator)
    interleaved = torch.stack([keys, values], dim=-1).reshape(batch_size, -1)
    query_indices = torch.randint(0, num_pairs, (batch_size,), generator=generator)
    query_key = keys[torch.arange(batch_size), query_indices].unsqueeze(1)
    answer_value = values[torch.arange(batch_size), query_indices].unsqueeze(1)
    query = torch.full((batch_size, 1), ASSOCIATIVE_QUERY, dtype=torch.long)
    answer = torch.full((batch_size, 1), ASSOCIATIVE_ANSWER, dtype=torch.long)
    tokens = torch.cat([interleaved, query, query_key, answer, answer_value], dim=1)
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    target_mask[:, -1:] = True
    return {"inputs": tokens, "targets": tokens.clone(), "target_mask": target_mask}


def generate_brackets_batch(
    batch_size: int,
    max_depth: int = 4,
    noise_tokens: int = 0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    sequences = []
    masks = []
    max_len = 0
    bracket_pairs = list(BRACKET_OPEN.items())
    for _ in range(batch_size):
        depth = int(rng.integers(1, max_depth + 1))
        stack = []
        sequence = []
        mask = []
        for _ in range(depth):
            open_token, close_token = bracket_pairs[int(rng.integers(0, len(bracket_pairs)))]
            sequence.append(open_token)
            mask.append(False)
            stack.append(close_token)
            if noise_tokens > 0 and rng.random() < 0.3:
                for _ in range(int(rng.integers(0, noise_tokens + 1))):
                    sequence.append(7 + int(rng.integers(0, 4)))
                    mask.append(False)
        while stack:
            sequence.append(stack.pop())
            mask.append(True)
        max_len = max(max_len, len(sequence))
        sequences.append(sequence)
        masks.append(mask)
    padded = []
    padded_masks = []
    for sequence, mask in zip(sequences, masks):
        pad_len = max_len - len(sequence)
        padded.append(sequence + [BRACKET_PAD] * pad_len)
        padded_masks.append(mask + [False] * pad_len)
    tokens = torch.tensor(padded, dtype=torch.long)
    target_mask = torch.tensor(padded_masks, dtype=torch.bool)
    return {"inputs": tokens, "targets": tokens.clone(), "target_mask": target_mask}
