from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from models.base import BaseSequenceModel, LinearState
from utils.io import SequenceRecord
from utils.metrics import accuracy_score, aupr_score, auroc_score, bce_loss_from_logits, sigmoid


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_loss: float
    val_loss: float
    val_auroc: float
    val_aupr: float
    val_acc: float


def train_test_split_71515(records: Sequence[SequenceRecord], seed: int = 42):
    from utils.io import split_71515

    return split_71515(records, seed=seed)


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def evaluate_model(model: BaseSequenceModel, records: Sequence[SequenceRecord], pos_weight: float = 1.0) -> dict[str, float]:
    if not records:
        return {"loss": 0.0, "auroc": 0.5, "aupr": 0.0, "acc": 0.0}
    labels: List[int] = []
    probs: List[float] = []
    loss_total = 0.0
    for record in records:
        logit = model.predict_logit(record.on_seq, record.off_seq)
        prob = sigmoid(logit)
        labels.append(record.label)
        probs.append(prob)
        loss_total += bce_loss_from_logits(logit, record.label, pos_weight=pos_weight)
    return {
        "loss": loss_total / len(records),
        "auroc": auroc_score(labels, probs),
        "aupr": aupr_score(labels, probs),
        "acc": accuracy_score(labels, probs),
    }


def fit_linear_model(
    model: BaseSequenceModel,
    train_records: Sequence[SequenceRecord],
    val_records: Sequence[SequenceRecord],
    epochs: int = 8,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    pos_weight: float | None = None,
) -> tuple[BaseSequenceModel, List[TrainingHistoryEntry]]:
    rng = random.Random(seed)
    train_records = list(train_records)
    if pos_weight is None:
        positives = sum(1 for record in train_records if record.label == 1)
        negatives = max(0, len(train_records) - positives)
        pos_weight = min(1000.0, negatives / positives) if positives else 1.0
        if pos_weight <= 0:
            pos_weight = 1.0

    weights = list(model.state.weights)
    bias = float(model.state.bias)
    best_state = LinearState(bias=bias, weights=list(weights), metadata=dict(model.state.metadata))
    best_val_aupr = -1.0
    best_entry: TrainingHistoryEntry | None = None
    history: List[TrainingHistoryEntry] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_records)
        train_loss = 0.0
        for record in train_records:
            features = model.feature_vector(record.on_seq, record.off_seq)
            logit = bias + _dot(weights, features)
            prob = sigmoid(logit)
            label = record.label
            sample_weight = pos_weight if label == 1 else 1.0
            grad = (prob - label) * sample_weight
            train_loss += bce_loss_from_logits(logit, label, pos_weight=pos_weight)
            for i, feature in enumerate(features):
                weights[i] -= learning_rate * (grad * feature + weight_decay * weights[i])
            bias -= learning_rate * grad

        model.update_state(bias=bias, weights=weights, metadata=model.state.metadata)
        val_metrics = evaluate_model(model, val_records, pos_weight=pos_weight)
        entry = TrainingHistoryEntry(
            epoch=epoch,
            train_loss=train_loss / max(1, len(train_records)),
            val_loss=val_metrics["loss"],
            val_auroc=val_metrics["auroc"],
            val_aupr=val_metrics["aupr"],
            val_acc=val_metrics["acc"],
        )
        history.append(entry)
        if entry.val_aupr > best_val_aupr:
            best_val_aupr = entry.val_aupr
            best_state = LinearState(bias=bias, weights=list(weights), metadata=dict(model.state.metadata))
            best_entry = entry
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.update_state(bias=best_state.bias, weights=best_state.weights, metadata=best_state.metadata)
    return model, history


def serialize_history(history: Sequence[TrainingHistoryEntry]) -> list[dict[str, float]]:
    return [
        {
            "epoch": entry.epoch,
            "train_loss": float(entry.train_loss),
            "val_loss": float(entry.val_loss),
            "val_auroc": float(entry.val_auroc),
            "val_aupr": float(entry.val_aupr),
            "val_acc": float(entry.val_acc),
        }
        for entry in history
    ]
