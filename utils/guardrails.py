"""
AGENTS.md compliance: [guardrails=True, target=Codex/Kimi/Claude]
确认本文件遵守 AGENTS.md 约束

utils/guardrails.py — 代码级强制约束检查

用法：
    from utils.guardrails import check_model_config, check_eval_procedure

    check_model_config(config)
    check_eval_procedure("results/checkpoints/best.pt", checkpoint_type="best")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from numbers import Real
from pathlib import Path
from typing import Any


VALID_SPLIT_MODES = ("random", "sgrna_safe", "loo")
_MISSING = object()


class GuardrailsViolation(Exception):
    """违反 AGENTS.md 约束时抛出。"""

    def __init__(self, message: str, constraint_id: str | int | None = None):
        self.constraint_id = constraint_id
        full_msg = "🔴 AGENTS.md 约束违反"
        if constraint_id is not None:
            full_msg += f" (约束#{constraint_id})"
        full_msg += f":\n{message}\n"
        full_msg += "\n请阅读 AGENTS.md 了解约束详情和修复方法。"
        super().__init__(full_msg)


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return vars(config)
    try:
        return dict(config)
    except Exception as exc:  # pragma: no cover - defensive path
        raise GuardrailsViolation(
            "config 必须是 dict、dataclass、argparse.Namespace、SimpleNamespace "
            "或具有 __dict__ 属性的对象。",
            constraint_id="通用",
        ) from exc


def _lookup(config: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    if key in config:
        return config[key]

    # 支持常见嵌套配置：model.use_rnafm、training.split_mode、loss.pos_weight 等。
    for section_name in (
        "model",
        "rnafm",
        "data",
        "dataset",
        "training",
        "train",
        "loss",
        "evaluation",
        "eval",
        "runtime",
    ):
        section = config.get(section_name)
        if isinstance(section, Mapping) and key in section:
            return section[key]
        if hasattr(section, "__dict__"):
            section_vars = vars(section)
            if key in section_vars:
                return section_vars[key]

    return default


def _ensure_explicit_bool(
    value: Any,
    field_name: str,
    constraint_id: int,
) -> None:
    if value is _MISSING:
        raise GuardrailsViolation(
            f"config 中缺少 '{field_name}' 字段。必须显式声明为 true 或 false。",
            constraint_id=constraint_id,
        )
    if value is None or not isinstance(value, bool):
        raise GuardrailsViolation(
            f"'{field_name}' 必须显式声明为 true 或 false，当前值为 {value!r}。",
            constraint_id=constraint_id,
        )


def _numeric_pos_weight(pos_weight: Any) -> float:
    if isinstance(pos_weight, bool):
        raise GuardrailsViolation(
            f"pos_weight 必须是数值或 None，当前值为 {pos_weight!r}。",
            constraint_id=5,
        )
    if isinstance(pos_weight, Real):
        return float(pos_weight)
    if isinstance(pos_weight, str):
        try:
            return float(pos_weight)
        except ValueError as exc:
            raise GuardrailsViolation(
                f"pos_weight 必须是数值或 None，当前值为 {pos_weight!r}。",
                constraint_id=5,
            ) from exc
    raise GuardrailsViolation(
        f"pos_weight 必须是数值或 None，当前值为 {pos_weight!r}。",
        constraint_id=5,
    )


def check_model_config(config: Any) -> bool:
    """
    检查模型配置是否遵守 AGENTS.md 约束。

    必须在模型初始化或训练入口尽早调用。

    Args:
        config: dict / dataclass / argparse.Namespace / SimpleNamespace / 配置对象。

    Raises:
        GuardrailsViolation: 如果违反约束 #1/#2/#4/#5。
    """
    cfg = _as_mapping(config)

    # === 约束 #1：use_rnafm 必须显式声明 ===
    use_rnafm = _lookup(cfg, "use_rnafm")
    _ensure_explicit_bool(use_rnafm, "use_rnafm", constraint_id=1)

    # === 约束 #2：use_rnafm=true 时 freeze_rnafm 必须显式声明 ===
    freeze_rnafm = _lookup(cfg, "freeze_rnafm")
    if use_rnafm is True:
        _ensure_explicit_bool(freeze_rnafm, "freeze_rnafm", constraint_id=2)
    elif freeze_rnafm is not _MISSING and freeze_rnafm is not None:
        _ensure_explicit_bool(freeze_rnafm, "freeze_rnafm", constraint_id=2)

    # === 约束 #4：split_mode 必须显式声明且合法 ===
    split_mode = _lookup(cfg, "split_mode")
    if split_mode is _MISSING:
        raise GuardrailsViolation(
            "config 中缺少 'split_mode' 字段。\n"
            f"必须是以下之一: {list(VALID_SPLIT_MODES)}\n"
            "  random: 随机划分（仅 debug 或复现实验）\n"
            "  sgrna_safe: sgRNA-safe group split（推荐，真实性能）\n"
            "  loo: Leave-One-sgRNA-Out（最严格泛化测试）",
            constraint_id=4,
        )
    if split_mode not in VALID_SPLIT_MODES:
        raise GuardrailsViolation(
            f"split_mode={split_mode!r} 无效，必须是 {list(VALID_SPLIT_MODES)}。",
            constraint_id=4,
        )

    # === 约束 #5：pos_weight 上限为 50 ===
    pos_weight = _lookup(cfg, "pos_weight", default=None)
    if pos_weight is not None:
        pos_weight_value = _numeric_pos_weight(pos_weight)
        if pos_weight_value <= 0:
            raise GuardrailsViolation(
                f"pos_weight={pos_weight_value:g} 非法，必须为正数。",
                constraint_id=5,
            )
        if pos_weight_value > 50:
            n_neg = _lookup(cfg, "n_neg", default=1000)
            n_pos = _lookup(cfg, "n_pos", default=1)
            try:
                suggested = (float(n_neg) / max(float(n_pos), 1.0)) ** 0.5
            except Exception:
                suggested = 12.0
            raise GuardrailsViolation(
                f"pos_weight={pos_weight_value:g} 超过上限 50。\n"
                "大 pos_weight 可能导致 precision 崩溃和 AUPRC 解释失真。\n"
                "建议：\n"
                f"  1. 降低 pos_weight 至 sqrt(n_neg/n_pos) 级别，当前估计约 {suggested:.3g}\n"
                "  2. 改用 focal_loss (gamma=2.0)",
                constraint_id=5,
            )

    print(
        "✅ Guardrails 检查通过: "
        f"use_rnafm={use_rnafm}, "
        f"freeze_rnafm={None if freeze_rnafm is _MISSING else freeze_rnafm}, "
        f"split_mode={split_mode}, "
        f"pos_weight={pos_weight}"
    )
    return True


def check_eval_procedure(
    checkpoint_path: str | Path,
    checkpoint_type: str = "best",
    *,
    require_exists: bool = False,
) -> bool:
    """
    检查评估流程是否遵守 AGENTS.md 约束 #6。

    Test 评估必须使用 validation AUPRC 最佳 checkpoint。
    """
    checkpoint_path = Path(checkpoint_path)
    normalized_type = str(checkpoint_type).lower().strip()

    if normalized_type != "best":
        raise GuardrailsViolation(
            f"test 评估使用了 {checkpoint_type!r} checkpoint。\n"
            "AGENTS.md 约束 #6 要求 test 评估必须使用 'best' checkpoint "
            "（val AUPRC 最高时的模型）。\n"
            f"请显式加载 best.pt，例如: torch.load('{checkpoint_path.parent / 'best.pt'}')",
            constraint_id=6,
        )

    if "last" in checkpoint_path.name.lower():
        raise GuardrailsViolation(
            f"checkpoint_path 指向 last checkpoint: {checkpoint_path}\n"
            "test 评估必须显式加载 best.pt。",
            constraint_id=6,
        )

    if require_exists and not checkpoint_path.exists():
        raise GuardrailsViolation(
            f"未找到 checkpoint: {checkpoint_path}",
            constraint_id=6,
        )

    if not checkpoint_path.exists():
        best_path = checkpoint_path.parent / "best.pt"
        if best_path.exists():
            print(f"⚠️  未找到 {checkpoint_path}，但发现 {best_path}")
            print(f"    建议改用: {best_path}")
        else:
            print(f"⚠️  未找到 checkpoint: {checkpoint_path}")

    print(f"✅ 评估检查通过: 使用 {normalized_type} checkpoint ({checkpoint_path})")
    return True


def _infer_position_count(run_states: Any) -> int:
    if isinstance(run_states, int):
        return run_states
    if hasattr(run_states, "shape") and getattr(run_states, "shape"):
        return int(run_states.shape[0])
    if hasattr(run_states, "__len__"):
        return len(run_states)
    raise GuardrailsViolation(
        "无法判断 run_states 的位置数量；请传入 list、tuple、tensor、ndarray 或位置数 int。",
        constraint_id=3,
    )


def check_run_encoding_positions(
    run_states: Any,
    max_pos: int = 20,
    *,
    name: str = "run_states",
) -> bool:
    """
    检查 Run 编码是否只覆盖 positions 1-20。

    Args:
        run_states: Run 状态序列、tensor/ndarray，或直接传入位置数量 int。
        max_pos: 最大允许位置数，默认 20。
    """
    n_positions = _infer_position_count(run_states)
    if n_positions > max_pos:
        raise GuardrailsViolation(
            f"{name} 覆盖了 {n_positions} 个位置，超过上限 {max_pos}。\n"
            "AGENTS.md 约束 #3: PAM 位（positions 21-23）不参与 Run 状态。\n"
            f"请确保 compute_run_states() 只处理 positions 1-{max_pos}。",
            constraint_id=3,
        )
    return True


def report_metrics(
    auroc: float | None,
    auprc: float | None,
    split_mode: str | None,
) -> dict[str, float | str]:
    """
    报告指标，确保同时输出 AUROC、AUPRC 和 split_mode。

    === 约束 #7：AUPRC 和 AUROC 必须同时报告 ===
    """
    if auroc is None or auprc is None:
        raise GuardrailsViolation(
            "必须同时报告 AUROC 和 AUPRC，禁止只报告其中一个指标。",
            constraint_id=7,
        )
    if split_mode not in VALID_SPLIT_MODES:
        raise GuardrailsViolation(
            f"报告指标时必须注明合法 split_mode，当前为 {split_mode!r}。",
            constraint_id=7,
        )

    print("\n" + "=" * 50)
    print(f"Test Results (split_mode={split_mode}):")
    print(f"  AUROC: {auroc:.6f}")
    print(f"  AUPRC: {auprc:.6f}")
    print("=" * 50)

    if split_mode == "sgrna_safe" and auroc > 0.99 and auprc < 0.1:
        print("\n⚠️  警告: AUROC 很高但 AUPRC 很低。")
        print("    在 group-safe split 下这可能反映严格泛化难度，但请确认模型没有 underfit。")

    return {"AUROC": float(auroc), "AUPRC": float(auprc), "split_mode": split_mode}


def check_cclmoff_columns(columns: Any) -> bool:
    """
    检查 CCLMoff 字段假设是否符合 AGENTS.md 约束 #8。

    该函数不会要求所有元数据字段都存在；它提醒调用方 Method/Length 可能缺失，
    不得把 label=0 当作安全位点，也不得把 sgRNA_type 当作检测方法。
    """
    colset = set(columns)
    required_core = {"sgRNA_seq", "off_seq", "sgRNA_type", "label", "id"}
    missing_core = sorted(required_core - colset)
    if missing_core:
        raise GuardrailsViolation(
            f"CCLMoff 核心字段缺失: {missing_core}",
            constraint_id=8,
        )

    if "Method" not in colset:
        print("⚠️  CCLMoff 数据缺少 Method 字段：Tier-aware 训练前必须外部补充或显式标为 unknown。")
    if "Length" not in colset:
        print("⚠️  CCLMoff 数据缺少 Length 字段：不要假设派生表保留完整元数据。")

    return True
