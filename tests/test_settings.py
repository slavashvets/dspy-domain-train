import os
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from dspy_domain_train.settings import Settings
from dspy_domain_train.toml import config_dir

BASE_TOML = """
train_path = "data/train.json"
dev_path = "data/dev.json"
test_path = "data/test.json"
num_threads = 4
seed = 42

[gepa]
auto = "light"
"""

LOCAL_TOML = """
[eval]
endpoint = "https://eval.example.openai.azure.com/"
api_key = "eval-secret"
api_version = "v1"
deployment = "eval-deployment"
model_type = "chat"
temperature = 0.0

[refine]
endpoint = "https://refine.example.openai.azure.com/"
api_key = "refine-secret"
api_version = "v1"
deployment = "refine-deployment"
model_type = "chat"
temperature = 1.0
max_tokens = 16000
"""


class SettingsTomlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.cwd = Path.cwd()
        self.env = {
            key: value for key, value in os.environ.items() if key.startswith("DSPY_")
        }

        for key in self.env:
            os.environ.pop(key, None)
        os.environ["DSPY_PROFILE"] = "local"
        config_dir.cache_clear()
        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        config_dir.cache_clear()
        for key in list(os.environ):
            if key.startswith("DSPY_"):
                os.environ.pop(key, None)
        os.environ.update(self.env)
        self.temp_dir.cleanup()

    def write_settings(self, local: str = LOCAL_TOML) -> None:
        (self.root / "settings.toml").write_text(BASE_TOML, encoding="utf-8")
        (self.root / "settings.local.toml").write_text(local, encoding="utf-8")

    def test_loads_base_and_local_toml_with_relative_paths(self) -> None:
        self.write_settings()

        settings = Settings()  # type: ignore[call-arg]

        self.assertEqual(settings.gepa.auto, "light")
        self.assertEqual(settings.num_threads, 4)
        self.assertEqual(settings.eval.deployment, "eval-deployment")
        self.assertEqual(settings.refine.max_tokens, 16000)
        self.assertEqual(settings.eval.api_key.get_secret_value(), "eval-secret")
        self.assertEqual(settings.train_path, self.root / "data/train.json")
        self.assertEqual(settings.dev_path, self.root / "data/dev.json")
        self.assertEqual(settings.test_path, self.root / "data/test.json")

    def test_env_overrides_toml(self) -> None:
        self.write_settings()
        os.environ["DSPY_NUM_THREADS"] = "8"
        os.environ["DSPY_EVAL__DEPLOYMENT"] = "env-deployment"

        settings = Settings()  # type: ignore[call-arg]

        self.assertEqual(settings.num_threads, 8)
        self.assertEqual(settings.eval.deployment, "env-deployment")

    def test_rejects_unknown_toml_keys(self) -> None:
        self.write_settings(local=LOCAL_TOML + "\nunknown = true\n")

        with self.assertRaises(ValidationError):
            Settings()  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
