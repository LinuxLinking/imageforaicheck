import math
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..modules.ai_detector import AIDetector
from ..modules.object_detector import ObjectDetector


# 包含 YOLO 特征的完整特征列表
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
    # 新增 YOLO 特征
    "yolo_num_detections_n",
    "yolo_avg_confidence_n",
    "yolo_max_confidence_n",
    "yolo_std_confidence_n",
    "yolo_anomaly_score",
)

# 不带 YOLO 的特征（兼容旧版本）
NO_YOLO_FEATURE_NAMES = (
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
        # YOLO 特征
        "yolo_num_detections_n": float(normalized_features.get("yolo_num_detections", 0.0) or 0.0),
        "yolo_avg_confidence_n": float(normalized_features.get("yolo_avg_confidence", 0.0) or 0.0),
        "yolo_max_confidence_n": float(normalized_features.get("yolo_max_confidence", 0.0) or 0.0),
        "yolo_std_confidence_n": float(normalized_features.get("yolo_std_confidence", 0.0) or 0.0),
        "yolo_anomaly_score": float(normalized_features.get("yolo_anomaly_score", 0.0) or 0.0),
    }
    return [float(mapping.get(name, 0.0) or 0.0) for name in feature_names]


def extract_base_features(image_path: str, include_yolo: bool = True) -> Dict[str, Any]:
    """
    提取基础特征，可选包含 YOLO 特征
    """
    with Image.open(image_path) as img:
        width, height = img.size
        fmt = img.format

    detector = AIDetector()
    detected = detector.detect(image_path)
    normalized = detected.get("normalized_features", {}) or {}
    
    result = {
        "format": fmt,
        "width": int(width or 0),
        "height": int(height or 0),
        "ai_probability_raw": float(detected.get("ai_probability", 0.0) or 0.0),
        "confidence_raw": float(detected.get("confidence", 0.0) or 0.0),
        "normalized_features": normalized,
    }
    
    # 如果启用，添加 YOLO 特征
    if include_yolo:
        try:
            yolo_detector = ObjectDetector()
            yolo_features = yolo_detector.extract_yolo_features(image_path)
            
            if yolo_features and yolo_features.get('success'):
                # 把 YOLO 特征合并到 normalized_features 中
                norm_yolo = {
                    "yolo_num_detections": min(yolo_features.get('num_detections', 0) / 20.0, 1.0),
                    "yolo_avg_confidence": yolo_features.get('avg_confidence', 0.0),
                    "yolo_max_confidence": yolo_features.get('max_confidence', 0.0),
                    "yolo_std_confidence": yolo_features.get('std_confidence', 0.0),
                    "yolo_anomaly_score": yolo_features.get('yolo_anomaly_score', 0.0),
                }
                normalized.update(norm_yolo)
                result['yolo_features'] = yolo_features
        except Exception:
            pass
    
    return result


def extract_feature_vector(
    image_path: str,
    feature_names: Tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    include_yolo: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    base = extract_base_features(image_path, include_yolo=include_yolo)
    vector = build_feature_vector(
        base.get("normalized_features", {}) or {},
        base.get("width", 0),
        base.get("height", 0),
        feature_names=feature_names,
    )
    return vector, base

