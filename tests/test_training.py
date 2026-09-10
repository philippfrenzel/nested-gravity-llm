import unittest
from types import SimpleNamespace

import torch

from config import set_seed
from data.text_data import CharacterTextDataset
from data.synthetic import generate_associative_recall_batch, generate_brackets_batch, generate_copy_batch
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
                batch["targets_are_aligned"],
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

    def test_extra_metrics_cover_other_tasks(self):
        recall_batch = generate_associative_recall_batch(batch_size=2, num_pairs=4, vocab_size=32, seed=42)
        recall_logits = torch.randn(2, recall_batch["inputs"].shape[1], 32)
        recall_metrics = extra_task_metrics(
            SimpleNamespace(task="associative_recall", gap_length=0, num_pairs=4),
            recall_logits,
            recall_batch["targets"],
            recall_batch["target_mask"],
            recall_batch["targets_are_aligned"],
        )
        self.assertIn("accuracy_num_pairs_4", recall_metrics)

        bracket_batch = generate_brackets_batch(batch_size=2, max_depth=4, noise_tokens=1, seed=43)
        bracket_logits = torch.randn(2, bracket_batch["inputs"].shape[1], 16)
        bracket_metrics = extra_task_metrics(
            SimpleNamespace(task="brackets", gap_length=0, num_pairs=0),
            bracket_logits,
            bracket_batch["targets"],
            bracket_batch["target_mask"],
            bracket_batch["targets_are_aligned"],
        )
        self.assertIn("token_accuracy", bracket_metrics)

        text_dataset = CharacterTextDataset.from_path(None, sequence_length=16, seed=42)
        text_batch = text_dataset.sample_batch("train", batch_size=2, step=0)
        text_logits = torch.randn(2, text_batch["inputs"].shape[1], len(text_dataset.tokenizer.vocab))
        text_metrics = sequence_metrics(
            text_logits,
            text_batch["targets"],
            text_batch["target_mask"],
            text_batch["targets_are_aligned"],
        )
        self.assertTrue(torch.isfinite(text_metrics["loss"]))

        bracket_sequence_metrics = sequence_metrics(
            bracket_logits,
            bracket_batch["targets"],
            bracket_batch["target_mask"],
            bracket_batch["targets_are_aligned"],
        )
        self.assertTrue(torch.isfinite(bracket_sequence_metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
