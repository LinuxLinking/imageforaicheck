"""
YOLO 特征提取器
用于从 YOLO 检测结果中提取特征，融入 AI 生成检测
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from PIL import Image

from .object_detector import ObjectDetector


class YOLOFeatureExtractor:
    """YOLO 特征提取器"""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.object_detector = ObjectDetector(model_name)
        # 常见场景关键词
        self.scene_keywords = {
            "street": ["car", "person", "bus", "truck", "traffic light"],
            "forest": ["tree", "plant", "bird", "animal"],
            "indoor": ["chair", "table", "couch", "bed", "tv", "laptop"],
            "portrait": ["person"],
            "landscape": ["tree", "mountain", "water", "sky"],
        }

    def extract_yolo_features(self, image_path: str) -> Dict[str, Any]:
        """
        提取 YOLO 相关特征

        返回:
            - object_count: 检测到的物体数量
            - classes: 检测到的物体类别列表
            - class_distribution: 各类别数量分布
            - scene_type: 推测的场景类型
            - object_density: 物体密度
            - abnormal_score: AI 生成异常分数（基于物体检测）
            - consistency_score: 物体一致性分数
        """
        # 检测物体
        detection_result = self.object_detector.detect_objects(image_path)

        if not detection_result.get("success", False):
            return {
                "object_count": 0,
                "classes": [],
                "class_distribution": {},
                "scene_type": "unknown",
                "object_density": 0.0,
                "abnormal_score": 0.0,
                "consistency_score": 1.0,
            }

        detections = detection_result.get("detections", [])
        classes = detection_result.get("classes", [])

        # 计算类别分布
        class_distribution = {}
        for det in detections:
            cls = det.get("class", "unknown")
            class_distribution[cls] = class_distribution.get(cls, 0) + 1

        # 获取图像尺寸计算密度
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                area = width * height
                object_density = len(detections) / (area / 100000)  # 每百万像素的物体数
        except Exception:
            object_density = 0.0

        # 推测场景类型
        scene_type = self._infer_scene_type(classes)

        # 计算异常分数
        abnormal_score = self._calculate_abnormal_score(
            detections,
            class_distribution,
            scene_type,
            object_density,
        )

        # 计算一致性分数
        consistency_score = self._calculate_consistency_score(
            detections,
            class_distribution,
            scene_type,
        )

        return {
            "object_count": len(detections),
            "classes": classes,
            "class_distribution": class_distribution,
            "scene_type": scene_type,
            "object_density": object_density,
            "abnormal_score": abnormal_score,
            "consistency_score": consistency_score,
            "detections": detections,
        }

    def _infer_scene_type(self, detected_classes: List[str]) -> str:
        """根据检测到的物体推测场景类型"""
        if not detected_classes:
            return "unknown"

        scene_scores = {}
        for scene, keywords in self.scene_keywords.items():
            match_count = sum(1 for cls in detected_classes if cls in keywords)
            scene_scores[scene] = match_count

        if not scene_scores:
            return "unknown"

        best_scene = max(scene_scores.items(), key=lambda x: x[1])
        if best_scene[1] > 0:
            return best_scene[0]
        return "unknown"

    def _calculate_abnormal_score(
        self,
        detections: List[Dict],
        class_distribution: Dict[str, int],
        scene_type: str,
        object_density: float,
    ) -> float:
        """
        计算 AI 生成异常分数（基于 YOLO 检测结果）

        分数越高，越可能是 AI 生成
        """
        score = 0.0

        # 1. 物体数量异常检查
        if len(detections) == 0:
            # 完全没有检测到物体可能是异常（但也可能是抽象图片）
            score += 0.1

        # 2. 物体密度异常检查
        if object_density > 100:  # 密度过高
            score += 0.15
        elif object_density < 0.5 and scene_type != "unknown":
            # 应该有物体的场景但物体太少
            score += 0.1

        # 3. 物体一致性检查（例如：太多同类物体）
        for cls, count in class_distribution.items():
            if count > 5:  # 某个物体太多了
                score += 0.1

        # 4. 场景与物体匹配检查
        if scene_type in self.scene_keywords:
            expected_classes = self.scene_keywords[scene_type]
            match_count = sum(1 for cls in class_distribution if cls in expected_classes)
            if match_count == 0 and len(class_distribution) > 0:
                # 场景和物体完全不匹配
                score += 0.2

        return min(score, 1.0)

    def _calculate_consistency_score(
        self,
        detections: List[Dict],
        class_distribution: Dict[str, int],
        scene_type: str,
    ) -> float:
        """
        计算物体一致性分数

        分数越高，越可能是真实图片
        """
        score = 1.0

        # 如果没有检测结果，返回中性分
        if not detections:
            return 0.5

        # 1. 检查物体多样性（AI 生成可能缺乏多样性）
        diversity = len(class_distribution) / max(len(detections), 1)
        if diversity < 0.2 and len(detections) > 3:
            score -= 0.15

        # 2. 检查场景匹配度
        if scene_type in self.scene_keywords:
            expected_classes = self.scene_keywords[scene_type]
            match_count = sum(1 for cls in class_distribution if cls in expected_classes)
            match_ratio = match_count / max(len(class_distribution), 1)
            score += (match_ratio - 0.5) * 0.2

        # 3. 检查置信度分布（AI 生成可能有异常低置信度）
        confidences = [det.get("confidence", 0) for det in detections]
        avg_confidence = np.mean(confidences) if confidences else 0
        if avg_confidence < 0.4:
            score -= 0.1

        return max(min(score, 1.0), 0.0)


def integrate_yolo_features(
    base_probability: float,
    yolo_features: Dict[str, Any],
    yolo_weight: float = 0.2,
) -> float:
    """
    将 YOLO 特征融入 AI 概率计算

    参数:
        base_probability: 基础 AI 概率（来自其他特征）
        yolo_features: YOLO 提取的特征
        yolo_weight: YOLO 特征的权重

    返回:
        融合后的 AI 概率
    """
    abnormal_score = yolo_features.get("abnormal_score", 0.0)
    consistency_score = yolo_features.get("consistency_score", 0.5)

    # 计算 YOLO 贡献的概率调整
    # 异常分数越高，越可能是 AI；一致性越高，越不可能是 AI
    yolo_contribution = abnormal_score * (1 - consistency_score)

    # 加权融合
    final_probability = (
        base_probability * (1 - yolo_weight) +
        yolo_contribution * yolo_weight
    )

    return max(min(final_probability, 1.0), 0.0)

