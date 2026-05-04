from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from utils.metrics import sigmoid
from utils.sequence import risk_level_from_probability


@dataclass
class LinearState:
    bias: float = 0.0
    weights: List[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSequenceModel:
    model_name: str = "base"
    encoder_name: str = ""
    feature_names: List[str] = []

    def __init__(self, state: LinearState | None = None):
        self.state = state or LinearState(bias=0.0, weights=[0.0] * len(self.feature_names))
        if not self.state.weights:
            self.state.weights = [0.0] * len(self.feature_names)
        if len(self.state.weights) != len(self.feature_names):
            raise ValueError("weight vector does not match feature names")

    def feature_vector(self, on_seq: str, off_seq: str) -> List[float]:
        raise NotImplementedError

    def predict_logit(self, on_seq: str, off_seq: str) -> float:
        features = self.feature_vector(on_seq, off_seq)
        return self.logit_from_features(features)

    def predict_probability(self, on_seq: str, off_seq: str) -> float:
        return sigmoid(self.predict_logit(on_seq, off_seq))

    def predict(self, on_seq: str, off_seq: str) -> dict[str, Any]:
        probability = self.predict_probability(on_seq, off_seq)
        return {
            "off_target_prob": probability,
            "risk_level": risk_level_from_probability(probability),
            "model_used": self.model_name,
            "encoder_used": self.encoder_name,
        }

    def logit_from_features(self, features: Sequence[float]) -> float:
        score = self.state.bias
        for weight, feature in zip(self.state.weights, features):
            score += weight * feature
        return score

    def update_state(self, bias: float, weights: Sequence[float], metadata: dict[str, Any] | None = None) -> None:
        self.state = LinearState(bias=bias, weights=list(weights), metadata=metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "encoder_name": self.encoder_name,
            "feature_names": list(self.feature_names),
            "bias": self.state.bias,
            "weights": list(self.state.weights),
            "metadata": dict(self.state.metadata),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str | Path) -> LinearState:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return LinearState(
            bias=float(payload.get("bias", 0.0)),
            weights=[float(value) for value in payload.get("weights", [])],
            metadata=dict(payload.get("metadata", {})),
        )

