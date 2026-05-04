from __future__ import annotations

import math
from typing import Iterable, Sequence


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def bce_loss_from_logits(logit: float, label: int, pos_weight: float = 1.0) -> float:
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if logit >= 0:
        loss = math.log1p(math.exp(-logit)) + (1 - label) * logit
    else:
        loss = math.log1p(math.exp(logit)) - label * logit
    if label == 1:
        loss *= pos_weight
    return loss


def accuracy_score(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> float:
    if not y_true:
        return 0.0
    correct = 0
    for label, score in zip(y_true, y_score):
        pred = 1 if score >= threshold else 0
        if pred == label:
            correct += 1
    return correct / len(y_true)


def auroc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    pos = sum(1 for y in y_true if y == 1)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5

    paired = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    prev_score = None
    points = [(0.0, 0.0)]

    for score, label in paired:
        if prev_score is None or score != prev_score:
            points.append((fp / neg, tp / pos))
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    points.append((fp / neg, tp / pos))
    points.sort(key=lambda item: item[0])

    area = 0.0
    prev_fpr, prev_tpr = points[0]
    for fpr, tpr in points[1:]:
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return max(0.0, min(1.0, area))


def aupr_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    pos = sum(1 for y in y_true if y == 1)
    if pos == 0:
        return 0.0

    paired = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    prev_precision = 1.0
    area = 0.0
    idx = 0
    n = len(paired)

    while idx < n:
        score = paired[idx][0]
        while idx < n and paired[idx][0] == score:
            _, label = paired[idx]
            if label == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        recall = tp / pos
        precision = tp / (tp + fp) if tp + fp else 1.0
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall = recall
        prev_precision = precision

    return max(0.0, min(1.0, area))

