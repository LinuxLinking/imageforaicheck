import math
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..modules.ai_detector import AIDetector


DEFAULT_FEATURE_NAMES = (
    "color_entropy_n",
    "noise_level_n",
    "blockiness_n",
    "edge_quality_n",
    "color_abnormality_n",
    "texture_score_n",
    "metadata_score_n",
    "log1p_width",
    "log1p_height",
)


def build_feature_vector(
    normalized_features: Dict[str, Any],
    width: int,
    height: int,
    feature_names: Tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> List[float]:
    mapping = {
        "color_entropy_n": float(normalized_features.get("color_entropy", 0.0) or 0.0),
        "noise_level_n": float(normalized_features.get("noise_level", 0.0) or 0.0),
        "blockiness_n": float(normalized_features.get("blockiness", 0.0) or 0.0),
        "edge_quality_n": float(normalized_features.get("edge_quality", 0.0) or 0.0),
        "color_abnormality_n": float(normalized_features.get("color_abnormality", 0.0) or 0.0),
        "texture_score_n": float(normalized_features.get("texture_score", 0.0) or 0.0),
        "metadata_score_n": float(normalized_features.get("metadata_score", 0.0) or 0.0),
        "log1p_width": float(math.log1p(max(int(width or 0), 0))),
        "log1p_height": float(math.log1p(max(int(height or 0), 0))),
    }
    return [float(mapping.get(name, 0.0) or 0.0) for name in feature_names]


def extract_base_features(image_path: str) -> Dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
        fmt = img.format

    detector = AIDetector()
    detected = detector.detect(image_path)
    normalized = detected.get("normalized_features", {}) or {}
    return {
        "format": fmt,
        "width": int(width or 0),
        "height": int(height or 0),
        "ai_probability_raw": float(detected.get("ai_probability", 0.0) or 0.0),
        "confidence_raw": float(detected.get("confidence", 0.0) or 0.0),
        "normalized_features": normalized,
    }


def extract_feature_vector(
    image_path: str,
    feature_names: Tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> Tuple[List[float], Dict[str, Any]]:
    base = extract_base_features(image_path)
    vector = build_feature_vector(
        base.get("normalized_features", {}) or {},
        base.get("width", 0),
        base.get("height", 0),
        feature_names=feature_names,
    )
    return vector, base

