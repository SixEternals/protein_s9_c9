import unittest

import torch

from models.conmismatch9_torch import (
    ConMismatch9TorchConfig,
    ConMismatch9TorchModel,
    GatedFusionHead,
    ResidualAuxiliaryFusionHead,
)


class ConMismatch9AblationTests(unittest.TestCase):
    def test_supported_ablation_modes_run_forward(self):
        modes = [
            "full",
            "legacy_full",
            "run_attn_no_mask",
            "fusion_norm",
            "run_attn_no_mask_fusion_norm",
            "no_mi",
            "no_run_attn",
            "no_fusion",
            "only_cnn",
            "no_mi_no_run_attn",
        ]
        x = torch.zeros(2, 23, 9)
        for mode in modes:
            with self.subTest(mode=mode):
                model = ConMismatch9TorchModel(
                    ConMismatch9TorchConfig(
                        hidden_dim=16,
                        attn_heads=4,
                        attn_layers=1,
                        dropout=0.0,
                        ablation_mode=mode,
                    )
                )
                model.eval()
                with torch.no_grad():
                    y = model(x)
                self.assertEqual(tuple(y.shape), (2,))

    def test_full_uses_residual_auxiliary_fusion(self):
        model = ConMismatch9TorchModel(
            ConMismatch9TorchConfig(
                hidden_dim=16,
                attn_heads=4,
                attn_layers=1,
                dropout=0.0,
                ablation_mode="full",
            )
        )
        self.assertIsInstance(model.fusion, ResidualAuxiliaryFusionHead)

        legacy_model = ConMismatch9TorchModel(
            ConMismatch9TorchConfig(
                hidden_dim=16,
                attn_heads=4,
                attn_layers=1,
                dropout=0.0,
                ablation_mode="legacy_full",
            )
        )
        self.assertIsInstance(legacy_model.fusion, GatedFusionHead)

        x = torch.zeros(2, 23, 9)
        model.eval()
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2,))

    def test_full_warmstart_matches_only_cnn(self):
        torch.manual_seed(0)
        only_cnn = ConMismatch9TorchModel(
            ConMismatch9TorchConfig(
                hidden_dim=16,
                attn_heads=4,
                attn_layers=1,
                dropout=0.0,
                ablation_mode="only_cnn",
            )
        )
        full = ConMismatch9TorchModel(
            ConMismatch9TorchConfig(
                hidden_dim=16,
                attn_heads=4,
                attn_layers=1,
                dropout=0.0,
                ablation_mode="full",
            )
        )

        full.warmstart_main_path_from_checkpoint({"model_state_dict": only_cnn.state_dict()})
        self.assertIsInstance(full.fusion, ResidualAuxiliaryFusionHead)
        mi_scale, run_scale = full.fusion.auxiliary_scales()
        self.assertAlmostEqual(float(mi_scale.item()), 0.0, places=6)
        self.assertAlmostEqual(float(run_scale.item()), 0.0, places=6)

        x = torch.randn(2, 23, 9)
        only_cnn.eval()
        full.eval()
        with torch.no_grad():
            base_logits = only_cnn(x)
            full_logits = full(x)
        torch.testing.assert_close(full_logits, base_logits)

    def test_invalid_ablation_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            ConMismatch9TorchModel(ConMismatch9TorchConfig(ablation_mode="bad_mode"))


if __name__ == "__main__":
    unittest.main()
