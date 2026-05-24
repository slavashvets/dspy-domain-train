import json
import tempfile
import unittest
from pathlib import Path

from dspy_domain_train.training import load_examples


class LoadExamplesTests(unittest.TestCase):
    def test_preserves_hard_slice_metadata_without_making_it_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "examples.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "dialogue_context": "system asked where to go",
                            "turn": "the restaurant please",
                            "domains": ["taxi"],
                            "example_id": "dev:12",
                            "hard_category": "implicit_context",
                            "source_split": "dev",
                            "source_index": 12,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            examples = load_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].example_id, "dev:12")
        self.assertEqual(examples[0].hard_category, "implicit_context")
        self.assertEqual(
            dict(examples[0].inputs()),
            {
                "dialogue_context": "system asked where to go",
                "turn": "the restaurant please",
            },
        )


if __name__ == "__main__":
    unittest.main()
