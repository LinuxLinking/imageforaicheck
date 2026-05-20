import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import optimize


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


@dataclass(frozen=True)
class LogisticRegressionModel:
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    medium_threshold: float
    high_threshold: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_std = standardize_apply(x, self.mean, self.std)
        logits = x_std @ self.weights + self.bias
        return sigmoid(logits)

    def to_json_dict(self) -> Dict:
        return {
            "type": "logreg",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "medium_threshold": float(self.medium_threshold),
            "high_threshold": float(self.high_threshold),
        }

    @staticmethod
    def from_json_dict(payload: Dict) -> "LogisticRegressionModel":
        return LogisticRegressionModel(
            feature_names=tuple(payload["feature_names"]),
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
            weights=np.array(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            medium_threshold=float(payload.get("medium_threshold", 0.6)),
            high_threshold=float(payload.get("high_threshold", 0.85)),
        )


def _loss_and_grad(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> Tuple[float, np.ndarray]:
    w = params[:-1]
    b = params[-1]
    logits = x @ w + b
    p = sigmoid(logits)
    eps = 1e-9
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    loss += 0.5 * l2 * float(np.sum(w * w))

    grad_logits = (p - y) / x.shape[0]
    grad_w = x.T @ grad_logits + l2 * w
    grad_b = float(np.sum(grad_logits))
    grad = np.concatenate([grad_w, np.array([grad_b], dtype=np.float64)])
    return float(loss), grad


def fit_logreg(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    l2: float = 0.8,
    medium_threshold: float = 0.6,
    high_threshold: float = 0.85,
) -> LogisticRegressionModel:
    mean, std = standardize_fit(x)
    x_std = standardize_apply(x, mean, std)

    init = np.zeros(x_std.shape[1] + 1, dtype=np.float64)

    def f(params: np.ndarray) -> float:
        loss, _ = _loss_and_grad(params, x_std, y, l2=l2)
        return loss

    def g(params: np.ndarray) -> np.ndarray:
        _, grad = _loss_and_grad(params, x_std, y, l2=l2)
        return grad

    result = optimize.minimize(
        f,
        init,
        jac=g,
        method="L-BFGS-B",
        options={"maxiter": 250},
    )

    params = result.x
    weights = params[:-1]
    bias = float(params[-1])

    return LogisticRegressionModel(
        feature_names=tuple(feature_names),
        mean=mean,
        std=std,
        weights=weights,
        bias=bias,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = max(int(y_true.shape[0]), 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / total
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def suggest_threshold_for_high_ratio(y_prob: np.ndarray, target_ratio: float) -> float:
    target_ratio = min(max(float(target_ratio), 0.01), 0.99)
    q = 1.0 - target_ratio
    return float(np.quantile(y_prob, q))

