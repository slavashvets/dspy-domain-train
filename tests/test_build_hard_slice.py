import unittest

from scripts.build_hard_slice import CATEGORIES, build_split


class BuildHardSliceTests(unittest.TestCase):
    def test_build_split_adds_stable_ids_and_balanced_categories(self) -> None:
        examples = [
            {
                "dialogue_context": "",
                "turn": "book a hotel and train",
                "domains": ["hotel", "train"],
            },
            {
                "dialogue_context": "Need a ride?",
                "turn": "yes please",
                "domains": ["taxi"],
            },
            {
                "dialogue_context": "Anything else?",
                "turn": "no thanks",
                "domains": ["none"],
            },
            {
                "dialogue_context": "Do you need the address?",
                "turn": "yes",
                "domains": ["restaurant"],
            },
            {
                "dialogue_context": "Can I book it?",
                "turn": "for monday at 12:00",
                "domains": ["restaurant"],
            },
            {
                "dialogue_context": "Do you prefer the north side?",
                "turn": "cheap please",
                "domains": ["hotel"],
            },
        ]

        hard = build_split(
            examples,
            split="train",
            per_category=1,
            seed=7,
        )

        self.assertEqual(len(hard), len(CATEGORIES))
        self.assertEqual({item["hard_category"] for item in hard}, set(CATEGORIES))
        self.assertTrue(all(item["example_id"].startswith("train:") for item in hard))


if __name__ == "__main__":
    unittest.main()
