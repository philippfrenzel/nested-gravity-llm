import unittest

import torch

from models import CausalTransformerLM, GRULanguageModel, NestedGravitationalLM


class CausalityTests(unittest.TestCase):
    def test_all_models_are_causal(self):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        changed = tokens.clone()
        changed[0, 6] = 9
        models = [
            NestedGravitationalLM(vocab_size=16, embedding_dim=32, hidden_dim=32, gravity_dim=8, num_centers=4, local_window=4),
            NestedGravitationalLM(
                vocab_size=16,
                embedding_dim=32,
                hidden_dim=32,
                gravity_dim=8,
                num_centers=4,
                num_parent_centers=2,
                local_window=4,
                use_nested_bridges=True,
            ),
            GRULanguageModel(vocab_size=16, embedding_dim=32, hidden_dim=32),
            CausalTransformerLM(vocab_size=16, embedding_dim=32, hidden_dim=32, max_sequence_length=16),
        ]
        for model in models:
            model.eval()
            with torch.no_grad():
                baseline = model(tokens)
                modified = model(changed)
            difference = (baseline[:, :6] - modified[:, :6]).abs().max().item()
            self.assertLess(difference, 1e-6, msg=model.__class__.__name__)


if __name__ == "__main__":
    unittest.main()
