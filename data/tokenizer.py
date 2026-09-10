from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharacterTokenizer:
    vocab: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        vocab = sorted(set(text))
        return cls(vocab=vocab)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: idx for idx, ch in enumerate(self.vocab)}

    @property
    def itos(self) -> dict[int, str]:
        return {idx: ch for idx, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        mapping = self.stoi
        return [mapping[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        inverse = self.itos
        return "".join(inverse[token] for token in token_ids)
