import unittest
from types import SimpleNamespace

import torch

from config import set_seed
from data.synthetic import generate_copy_batch
from models import NestedGravitationalLM
from train import extra_task_metrics, sequence_metrics, train_or_eval_epoch


class TrainingTests(unittest.TestCase):
    def test_nested_gravity_training_reduces_loss(self):
        set_seed(42)
        model = NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        masked_accuracies = []
        for step in range(8):
            batch = generate_copy_batch(batch_size=16, memory_length=4, gap_length=8, vocab_size=16, seed=42 + step)
            logits = model(batch["inputs"])
            metrics = sequence_metrics(logits, batch["targets"], batch["target_mask"])
            loss = metrics["loss"]
            masked_accuracy = extra_task_metrics(
                SimpleNamespace(task="copy", gap_length=8, num_pairs=0),
                logits,
                batch["targets"],
                batch["target_mask"],
            )["accuracy_at_gap_8"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            masked_accuracies.append(masked_accuracy)
        self.assertLess(sum(losses[-4:]) / 4.0, sum(losses[:4]) / 4.0)
        self.assertGreaterEqual(sum(masked_accuracies[-4:]) / 4.0, sum(masked_accuracies[:4]) / 4.0)

    def test_epoch_helper_trains_on_multiple_batches(self):
        set_seed(42)
        model = NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = SimpleNamespace(
            task="copy",
            batch_size=8,
            memory_length=4,
            gap_length=8,
            vocab_size=16,
            seed=42,
            num_pairs=0,
            max_depth=0,
            noise_tokens=0,
            steps_per_epoch=3,
            val_steps=1,
            center_loss_weight=0.0,
            repulsion_loss_weight=0.0,
            entropy_loss_weight=0.0,
            gradient_clip_norm=1.0,
        )
        metrics = train_or_eval_epoch(config, model, optimizer, "train")
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater(metrics["gradient_norm"], 0.0)
        self.assertIn("accuracy_at_gap_8", metrics)

        eval_metrics = train_or_eval_epoch(config, model, None, "val")
        self.assertTrue(torch.isfinite(torch.tensor(eval_metrics["loss"])))
        self.assertEqual(eval_metrics["gradient_norm"], 0.0)
        self.assertIn("accuracy_at_gap_8", eval_metrics)


if __name__ == "__main__":
    unittest.main()
