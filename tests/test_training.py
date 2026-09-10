import unittest

import torch
import torch.nn.functional as F

from config import set_seed
from data.synthetic import generate_copy_batch
from models import NestedGravitationalLM


class TrainingTests(unittest.TestCase):
    def test_nested_gravity_training_reduces_loss(self):
        set_seed(42)
        batch = generate_copy_batch(batch_size=16, memory_length=4, gap_length=8, vocab_size=16, seed=42)
        model = NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(8):
            logits = model(batch["inputs"])
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 16), batch["targets"][:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
