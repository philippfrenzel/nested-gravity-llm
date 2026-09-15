import unittest

import torch

from models import CausalTransformerLM, GRULanguageModel, NestedGravitationalLM


class ShapeTests(unittest.TestCase):
    def test_output_shapes(self):
        batch = torch.randint(0, 16, (4, 12))
        models = [
            NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4),
            GRULanguageModel(vocab_size=16, embedding_dim=32, hidden_dim=32),
            CausalTransformerLM(vocab_size=16, embedding_dim=32, hidden_dim=32, max_sequence_length=16),
        ]
        for model in models:
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (4, 12, 16))

    def test_nested_bridges_add_hierarchical_force(self):
        model = NestedGravitationalLM(
            vocab_size=16,
            embedding_dim=32,
            hidden_dim=32,
            gravity_dim=8,
            num_centers=8,
            num_parent_centers=4,
            num_universe_levels=4,
            universe_shrink_factor=2.0,
            local_window=4,
            use_nested_bridges=True,
        )

        logits, metrics = model(torch.randint(0, 16, (2, 12)), return_metrics=True)

        self.assertEqual(tuple(logits.shape), (2, 12, 16))
        self.assertGreater(metrics["mean_bridge_force_norm"], 0.0)
        self.assertGreater(metrics["parent_center_entropy"], 0.0)
        self.assertEqual(metrics["universe_center_counts"], [8, 4, 2, 1])
        self.assertGreaterEqual(metrics["active_universe_level"], 1.0)
        self.assertLessEqual(metrics["active_universe_level"], 4.0)

    def test_dense_universe_switches_upward(self):
        model = NestedGravitationalLM(
            vocab_size=16,
            num_centers=4,
            num_parent_centers=2,
            use_nested_bridges=True,
            universe_density_threshold=1.5,
            universe_switch_sharpness=4.0,
        )

        uniform_density, uniform_switch = model._density_switch(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
        dense_density, dense_switch = model._density_switch(torch.tensor([[7.0, 1.0, 0.0, 0.0]]))

        self.assertAlmostEqual(uniform_density.item(), 1.0)
        self.assertGreater(dense_density.item(), uniform_density.item())
        self.assertGreater(dense_switch.item(), uniform_switch.item())


if __name__ == "__main__":
    unittest.main()
