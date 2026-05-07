"""Lightweight tests that do NOT require torch.

Can be run on any Python environment with the project source:
    python -m unittest tests.test_light -v
"""

import tempfile
import unittest
from pathlib import Path

from server import _safe_static_path, ModelRegistry
from utils.config import load_config, save_config, validate_config


class TestStaticPathSafety(unittest.TestCase):
    def test_safe_file_inside_static(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (static_dir / "test.txt").write_text("hello")
            # Temporarily override cwd so Path("static") resolves to tmpdir
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _safe_static_path("/static/test.txt")
                self.assertIsNotNone(result)
                self.assertEqual(result.read_text(), "hello")
            finally:
                os.chdir(old_cwd)

    def test_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (Path(tmpdir) / "secret.txt").write_text("secret")
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _safe_static_path("/static/../secret.txt")
                self.assertIsNone(result)
            finally:
                os.chdir(old_cwd)

    def test_url_encoded_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (Path(tmpdir) / "secret.txt").write_text("secret")
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _safe_static_path("/static/%2e%2e%2fsecret.txt")
                self.assertIsNone(result)
            finally:
                os.chdir(old_cwd)

    def test_directory_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (static_dir / "subdir").mkdir()
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _safe_static_path("/static/subdir")
                self.assertIsNone(result)
            finally:
                os.chdir(old_cwd)

    def test_nonexistent_file_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _safe_static_path("/static/nonexistent.txt")
                self.assertIsNone(result)
            finally:
                os.chdir(old_cwd)


class TestConfigValidation(unittest.TestCase):
    def test_valid_minimal_config_file_not_found(self):
        errors = validate_config(
            {"model": "deepfocus", "encoder": "r9", "dataset_files": ["/nonexistent.npz"]}
        )
        self.assertTrue(any("not found" in e for e in errors))

    def test_missing_model(self):
        errors = validate_config({"encoder": "r9", "dataset_files": ["/tmp/x.npz"]})
        self.assertTrue(any("missing required field: model" in e for e in errors))

    def test_invalid_model(self):
        errors = validate_config({"model": "foo", "encoder": "r9", "dataset_files": ["/tmp/x.npz"]})
        self.assertTrue(any("invalid model" in e for e in errors))

    def test_model_encoder_mismatch(self):
        errors = validate_config(
            {"model": "deepfocus", "encoder": "c9", "dataset_files": ["/tmp/x.npz"]}
        )
        self.assertTrue(any("expects encoder" in e for e in errors))

    def test_missing_dataset(self):
        errors = validate_config({"model": "deepfocus", "encoder": "r9"})
        self.assertTrue(any("dataset_files" in e for e in errors))

    def test_invalid_training_field(self):
        errors = validate_config(
            {
                "model": "deepfocus",
                "encoder": "r9",
                "dataset_files": ["/tmp/x.npz"],
                "training": {"epochs": "thirty"},
            }
        )
        self.assertTrue(any("training.epochs" in e for e in errors))

    def test_empty_yaml_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_yaml_with_comments(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("# comment\nmodel: deepfocus\n")
            path = f.name
        try:
            config = load_config(path)
            self.assertEqual(config["model"], "deepfocus")
        finally:
            Path(path).unlink(missing_ok=True)


class TestServerHealthWithoutWeights(unittest.TestCase):
    def test_no_weights_degraded(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        health = registry.health()
        self.assertEqual(health["status"], "degraded")
        self.assertTrue(health["fallback_active"])
        self.assertEqual(health["loaded_models"], [])

    def test_list_models_marks_loaded_false(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        models = registry.list_models()["models"]
        self.assertEqual(len(models), 2)
        for m in models:
            self.assertFalse(m["loaded"])
            self.assertIsNone(m["loaded_from"])
            self.assertEqual(m["backend"], "legacy_json")


class TestPredict23ntValidation(unittest.TestCase):
    def test_20nt_rejected(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with self.assertRaises(ValueError) as ctx:
            registry.predict("deepfocus", "r9", "GAGTCCGAGCAGAAGAAGAA", "GAGTCCGAGCAGAAGAAGAA")
        self.assertIn("length must be 23", str(ctx.exception))

    def test_24nt_rejected(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with self.assertRaises(ValueError) as ctx:
            registry.predict("deepfocus", "r9", "GAGTCCGAGCAGAAGAAGAAGAAA", "GAGTCCGAGCAGAAGAAGAAGAAA")
        self.assertIn("length must be 23", str(ctx.exception))

    def test_invalid_char_rejected(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with self.assertRaises(ValueError) as ctx:
            registry.predict("deepfocus", "r9", "XXXXXXXXXXXXXXXXXXXXXXX", "XXXXXXXXXXXXXXXXXXXXXXX")
        self.assertIn("invalid characters", str(ctx.exception))

    def test_empty_rejected(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with self.assertRaises(ValueError) as ctx:
            registry.predict("deepfocus", "r9", "", "")
        self.assertIn("empty", str(ctx.exception))

    def test_valid_23nt_accepted(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        result = registry.predict("deepfocus", "r9", "GAGTCCGAGCAGAAGAAGAAGAA", "GAGTCCGAGCAGAAGAAGAAGAA")
        self.assertIn("off_target_prob", result)
        self.assertEqual(result["model_backend"], "legacy_json")

    def test_u_converted_to_t(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        # U is allowed and converted to T internally; should not raise
        result = registry.predict("deepfocus", "r9", "U" * 23, "A" * 23)
        self.assertIn("off_target_prob", result)


class TestPredictFileErrors(unittest.TestCase):
    def test_empty_sequences_reported(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        content = "sgRNA,dna\n,TTTTTTTTTTTTTTTTTTTTTTT\nAAAAAAAAAAAAAAAAAAAAAAA,\n"
        result = registry.predict_from_file("deepfocus", "r9", content, "csv", True)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["failed"], 2)
        errors = result["errors"]
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("sgRNA sequence is empty" in e["error"] for e in errors))
        self.assertTrue(any("dna sequence is empty" in e["error"] for e in errors))

    def test_bad_length_reported(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        content = "sgRNA,dna\nAAAA,TTTT\n"
        result = registry.predict_from_file("deepfocus", "r9", content, "csv", True)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertIn("length", result["errors"][0]["error"])

    def test_20nt_in_file_rejected(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        content = "sgRNA,dna\nGAGTCCGAGCAGAAGAAGAA,GAGTCCGAGCAGAAGAAGAA\n"
        result = registry.predict_from_file("deepfocus", "r9", content, "csv", True)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertIn("length", result["errors"][0]["error"])


class TestJobRegistry(unittest.TestCase):
    def test_submit_job_returns_queued(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"model": "deepfocus", "encoder": "r9", "dataset_files": ["/tmp/x.npz"]}')
            path = f.name
        try:
            job = registry.submit_train_job(path, temp_config_path=False)
            self.assertEqual(job["status"], "queued")
            self.assertIn("job_id", job)

            status = registry.get_job(job["job_id"])
            self.assertIn(status["status"], {"queued", "running", "failed"})
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_job_raises(self):
        registry = ModelRegistry(config_paths=[], device="cpu")
        with self.assertRaises(ValueError):
            registry.get_job("deadbeef")


if __name__ == "__main__":
    unittest.main()
