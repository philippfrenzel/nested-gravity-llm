import unittest

import torch

from config import set_seed
from data.synthetic import generate_copy_batch
from models import NestedGravitationalLM
from train import extra_task_metrics, sequence_metrics


class TrainingTests(unittest.TestCase):
    def test_nested_gravity_training_reduces_loss(self):
        set_seed(42)
        batch = generate_copy_batch(batch_size=16, memory_length=4, gap_length=8, vocab_size=16, seed=42)
        model = NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        masked_accuracies = []
        for _ in range(8):
            logits = model(batch["inputs"])
            metrics = sequence_metrics(logits, batch["targets"], batch["target_mask"])
            loss = metrics["loss"]
            masked_accuracy = extra_task_metrics(
                type("Config", (), {"task": "copy", "gap_length": 8, "num_pairs": 0})(),
                logits,
                batch["targets"],
                batch["target_mask"],
            )["accuracy_at_gap_8"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            masked_accuracies.append(masked_accuracy)
        self.assertLess(losses[-1], losses[0])
        self.assertGreaterEqual(masked_accuracies[-1], masked_accuracies[0])


if __name__ == "__main__":
    unittest.main()
