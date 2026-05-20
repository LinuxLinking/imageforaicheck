import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from ..ml.feature_extractor import DEFAULT_FEATURE_NAMES, build_feature_vector
from ..ml.logreg import LogisticRegressionModel


class MLClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model: LogisticRegressionModel | None = None

    def _load(self) -> LogisticRegressionModel:
        if self._model is not None:
            return self._model
        with open(self.model_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._model = LogisticRegressionModel.from_json_dict(payload)
        return self._model

    def is_available(self) -> bool:
        return bool(self.model_path and os.path.exists(self.model_path))

    def predict_probability(self, normalized_features: Dict[str, Any], width: int, height: int) -> float:
        model = self._load()
        vec = build_feature_vector(
            normalized_features=normalized_features or {},
            width=width,
            height=height,
            feature_names=tuple(model.feature_names) if model.feature_names else DEFAULT_FEATURE_NAMES,
        )
        x = np.array([vec], dtype=np.float64)
        prob = float(model.predict_proba(x)[0])
        return prob

    def thresholds(self) -> Tuple[float, float]:
        model = self._load()
        return float(model.medium_threshold), float(model.high_threshold)

