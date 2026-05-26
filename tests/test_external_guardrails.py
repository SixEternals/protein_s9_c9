from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.preflight_guardrails import run_preflight


def write_config(path: Path) -> Path:
    path.write_text("dummy: true\n", encoding="utf-8")
    return path


def write_manifest(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


class ExternalGuardrailsTest(unittest.TestCase):
    def test_bl35_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.yaml")
            manifest = write_manifest(
                root / "manifest.yaml",
                """
                task_id: bl35_ok
                agent: codex
                stage: train
                bl_version: BL3.5-Full
                architecture_layer: middleware
                config_path: config.yaml
                command: [python, scripts/train_bl3.py, --config, configs/bl3_5_full_cclmoff.yaml]
                policy:
                  use_rnafm: false
                  freeze_rnafm: null
                  split_mode: sgrna_safe
                  pos_weight: 12
                midware:
                  use_c9: true
                  use_r9: true
                  fusion_mode: full
                  allow_concat_only: false
                eval:
                  checkpoint_type: best
                  report_metrics: [AUROC, AUPRC]
                """,
            )

            result = run_preflight(manifest, root=root)

        self.assertTrue(result.ok)

    def test_bl35_rejects_rnafm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.yaml")
            manifest = write_manifest(
                root / "manifest.yaml",
                """
                task_id: bl35_bad_rnafm
                agent: codex
                stage: train
                bl_version: BL3.5-Full
                architecture_layer: middleware
                config_path: config.yaml
                command: [python, scripts/train_bl3.py]
                policy:
                  use_rnafm: true
                  freeze_rnafm: true
                  split_mode: sgrna_safe
                  pos_weight: 12
                midware:
                  use_c9: true
                  use_r9: true
                  fusion_mode: full
                  allow_concat_only: false
                eval:
                  checkpoint_type: best
                  report_metrics: [AUROC, AUPRC]
                """,
            )

            result = run_preflight(manifest, root=root)

        self.assertFalse(result.ok)
        self.assertTrue(any("BL3/BL3.5" in finding.message for finding in result.errors))

    def test_mainline_concat_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.yaml")
            manifest = write_manifest(
                root / "manifest.yaml",
                """
                task_id: bl35_bad_concat
                agent: codex
                stage: train
                bl_version: BL3.5-Full
                architecture_layer: middleware
                config_path: config.yaml
                command: [python, scripts/train_bl3.py]
                policy:
                  use_rnafm: false
                  freeze_rnafm: null
                  split_mode: sgrna_safe
                  pos_weight: 12
                midware:
                  use_c9: true
                  use_r9: true
                  fusion_mode: concat
                  allow_concat_only: false
                eval:
                  checkpoint_type: best
                  report_metrics: [AUROC, AUPRC]
                """,
            )

            result = run_preflight(manifest, root=root)

        self.assertFalse(result.ok)
        self.assertTrue(any("dynamic fusion" in finding.message or "concat" in finding.message for finding in result.errors))

    def test_eval_rejects_non_best_checkpoint_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.yaml")
            manifest = write_manifest(
                root / "manifest.yaml",
                """
                task_id: eval_bad_checkpoint_type
                agent: codex
                stage: eval
                bl_version: BL4-Run-only
                architecture_layer: evaluation
                config_path: config.yaml
                command: [python, scripts/train_bl3.py, --eval-only]
                policy:
                  use_rnafm: true
                  freeze_rnafm: true
                  split_mode: sgrna_safe
                  pos_weight: 12
                eval:
                  checkpoint_path: results/bl4_runonly_cclmoff/checkpoints/best.pt
                  checkpoint_type: final_epoch
                  report_metrics: [AUROC, AUPRC]
                """,
            )

            result = run_preflight(manifest, root=root)

        self.assertFalse(result.ok)
        self.assertTrue(any("best" in finding.message or "checkpoint" in finding.message for finding in result.errors))


if __name__ == "__main__":
    unittest.main()
