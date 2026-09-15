import unittest
from pathlib import Path

from chat_app import TextGenerator


class ChatApplicationTests(unittest.TestCase):
    def test_text_checkpoint_generates_known_characters(self):
        generator = TextGenerator(
            Path("outputs/checkpoints/nested_gravity_text_char_best.pt"), temperature=0.8, max_new_tokens=4
        )
        completion = generator.generate("Der ")

        self.assertEqual(len(completion), 4)
        self.assertTrue(set(completion).issubset(set(generator.tokenizer.vocab)))

    def test_rejects_characters_outside_training_vocabulary(self):
        generator = TextGenerator(
            Path("outputs/checkpoints/nested_gravity_text_char_best.pt"), temperature=0.8, max_new_tokens=1
        )

        with self.assertRaisesRegex(ValueError, "Unsupported characters"):
            generator.generate("hello @")


if __name__ == "__main__":
    unittest.main()