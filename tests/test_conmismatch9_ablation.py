import unittest

import torch

from models.conmismatch9_torch import ConMismatch9TorchConfig, ConMismatch9TorchModel


class ConMismatch9AblationTests(unittest.TestCase):
    def test_supported_ablation_modes_run_forward(self):
        modes = [
            "full",
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

    def test_invalid_ablation_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            ConMismatch9TorchModel(ConMismatch9TorchConfig(ablation_mode="bad_mode"))


if __name__ == "__main__":
    unittest.main()
