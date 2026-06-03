#!/usr/bin/env python3
"""
Export formal-split test predictions for BL5-v4-NoPAM-control from best.pt.

AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None,
                       focal_loss=True]
确认本文件遵守 AGENTS.md 约束：使用 formal sgRNA-safe split，
显式加载 best.pt 做 test prediction 导出，并同时保留 PAM positions 21-23
的原始字段用于后续审计；不训练新模型。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bl5_dynamic_fusion import BL5RunOnlyDynamicFusion
from scripts.train_bl5 import (
    BL5Arrays,
    BL5Dataset,
    SequentialDistributedSampler,
    formal_group_json_split,
    make_live_collate,
    predict_probabilities,
    setup_distributed,
    write_test_predictions,
    is_main_process,
)
from utils.config import load_config
from utils.guardrails import check_eval_procedure, check_model_config
from utils.rnafm import load_rnafm


def main():
    config = load_config("configs/bl5_v4_nopam_control.yaml")
    check_model_config(config)
    dist_info = setup_distributed()
    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )

    arrays = BL5Arrays(config)
    data_cfg = config.get("data", {})
    csv_path = data_cfg.get("cclmoff_csv")
    group_col = data_cfg.get("group_column", "sgRNA_type")

    import pandas as pd
    group_labels = pd.read_csv(csv_path, usecols=[group_col])[group_col].values
    if len(group_labels) != len(arrays.labels):
        raise ValueError("group label count mismatch")

    split_cfg = config.get("split", {})
    if split_cfg.get("strategy") != "formal_group_json":
        raise ValueError("NoPAM export requires split.strategy=formal_group_json")
    split_result = formal_group_json_split(
        arrays.labels.astype(np.int64),
        group_labels,
        split_cfg,
    )
    split_indices = {
        "train": split_result["train"],
        "val": split_result["val"],
        "test": split_result["test"],
    }

    test_dataset = BL5Dataset(arrays, split_indices["test"])

    rnafm_cfg = config.get("rnafm", {})
    rnafm_model, alphabet = load_rnafm(
        rnafm_cfg.get("checkpoint_path"), trust_local_checkpoint=True
    )
    for name, param in rnafm_model.named_parameters():
        if "contact_head" in name or "lm_head" in name:
            param.requires_grad = False

    model = BL5RunOnlyDynamicFusion(
        rnafm_model=rnafm_model,
        padding_idx=alphabet.padding_idx,
        config=config,
    ).to(device)

    ckpt_path = Path(config.get("output_dir", "results/BL5-v4-NoPAM-control")) / "checkpoints" / "best.pt"
    check_eval_procedure(ckpt_path, checkpoint_type="best", require_exists=True)
    ckpt = torch.load(
        ckpt_path, map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])

    if dist_info["distributed"]:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=False,
        )

    batch_size = int(config["training"]["batch_size"]) * 2
    num_workers = int(config["training"]["num_workers"])
    use_learnable_run = bool(config["model"].get("use_learnable_run", False))
    use_pam_encoder = bool(config["model"].get("use_pam_encoder", False))

    test_sampler = SequentialDistributedSampler(
        test_dataset, dist_info["rank"], dist_info["world_size"]
    )
    collate_fn = make_live_collate(
        alphabet,
        use_learnable_run=use_learnable_run,
        use_pam_encoder=use_pam_encoder,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    probs = predict_probabilities(model, test_loader, device, dist_info)

    if is_main_process(dist_info):
        output_path = Path(config.get("output_dir", "results/BL5-v4-NoPAM-control")) / "test_predictions.csv"
        write_test_predictions(
            csv_path,
            split_indices["test"],
            probs,
            output_path,
            split_name="test",
        )
        print(f"Exported: {output_path} ({len(probs)} rows)")
    else:
        print("Non-main rank, skipping CSV write.")


if __name__ == "__main__":
    main()
