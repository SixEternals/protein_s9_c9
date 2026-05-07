"""Tests for GPT-5.5 audit fixes that require torch.

Skipped automatically if torch is not installed.
"""

import tempfile
import unittest
from pathlib import Path

from server import ModelRegistry
from utils.config import load_config, save_config

try:
    from models.deepfocus_torch import DeepFocusTorchConfig, DeepFocusTorchModel

    _HAS_DEEPFOCUS_TORCH = True
except Exception:
    _HAS_DEEPFOCUS_TORCH = False


@unittest.skipUnless(_HAS_DEEPFOCUS_TORCH, "torch not available")
class TestDeepFocusAblationValidation(unittest.TestCase):
    def test_full_mode_accepted(self):
        model = DeepFocusTorchModel(DeepFocusTorchConfig(ablation_mode="full"))
        self.assertIsNotNone(model.transformer)

    def test_inception_only_mode_accepted(self):
        model = DeepFocusTorchModel(DeepFocusTorchConfig(ablation_mode="inception_only"))
        self.assertIsNone(model.transformer)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            DeepFocusTorchModel(DeepFocusTorchConfig(ablation_mode="bad_mode"))
        self.assertIn("unsupported DeepFocus ablation_mode", str(ctx.exception))

    def test_typo_rejected(self):
        with self.assertRaises(ValueError):
            DeepFocusTorchModel(DeepFocusTorchConfig(ablation_mode="ful"))

    def test_dash_normalized_invalid_still_rejected(self):
        """Dashes are normalized to underscores, but invalid modes still fail."""
        with self.assertRaises(ValueError):
            DeepFocusTorchModel(DeepFocusTorchConfig(ablation_mode="bad-mode"))


class TestYamlConfigLoading(unittest.TestCase):
    def test_load_json_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"model": "deepfocus", "encoder": "r9"}')
            path = f.name
        try:
            config = load_config(path)
            self.assertEqual(config["model"], "deepfocus")
            self.assertEqual(config["encoder"], "r9")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_yaml_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: conmismatch9\nencoder: c9\n")
            path = f.name
        try:
            config = load_config(path)
            self.assertEqual(config["model"], "conmismatch9")
            self.assertEqual(config["encoder"], "c9")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_and_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            payload = {"model": "deepfocus", "encoder": "r9", "nested": {"a": 1}}
            save_config(path, payload)
            loaded = load_config(path)
            self.assertEqual(loaded["model"], "deepfocus")
            self.assertEqual(loaded["nested"]["a"], 1)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
