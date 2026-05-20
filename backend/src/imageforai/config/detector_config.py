"""
AI 检测器配置文件
在这里修改你的参数和公式权重，不需要改代码！
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class FeatureWeights:
    """特征权重配置"""
    color_entropy: float = 0.15
    noise_level: float = 0.15
    blockiness: float = 0.15
    edge_quality: float = 0.15
    color_abnormality: float = 0.15
    texture_score: float = 0.15
    metadata_score: float = 0.10


@dataclass
class NormalizationParams:
    """特征归一化参数"""
    color_entropy_min: float = 2.0
    color_entropy_max: float = 6.0
    noise_level_min_log: float = 2.0
    noise_level_max_log: float = 7.6
    blockiness_min_log: float = 3.2
    blockiness_max_log: float = 7.6
    edge_quality_min_log: float = 4.0
    edge_quality_max_log: float = 8.6
    color_abnormality_min_log: float = 2.4
    color_abnormality_max_log: float = 5.6
    texture_score_min_log: float = 4.8
    texture_score_max_log: float = 7.0


@dataclass
class PenaltyParams:
    """惩罚项参数"""
    document_penalty_base: float = 0.28
    document_penalty_low_diversity: float = 0.08
    document_penalty_low_entropy: float = 0.06
    document_penalty_high_correlation: float = 0.05
    camera_penalty_base: float = 0.12
    camera_penalty_noise_range: float = 0.03
    screenshot_penalty: float = 0.18


@dataclass
class BoostParams:
    """增强项参数"""
    metadata_boost_base: float = 0.25
    metadata_boost_score_weight: float = 0.15


@dataclass
class ThresholdParams:
    """阈值参数"""
    ai_probability_threshold: float = 0.7
    confidence_threshold: float = 0.5
    medium_risk_threshold: float = 0.4
    high_risk_threshold: float = 0.75


@dataclass
class YOLOIntegrationConfig:
    """YOLO 集成配置"""
    enabled: bool = False
    object_detection_weight: float = 0.20
    # AI 生成图片中常见的异常检测模式
    abnormal_object_patterns: Dict[str, float] = field(default_factory=lambda: {
        "extra_limbs": 0.8,  # 多余肢体
        "weird_faces": 0.7,  # 奇怪人脸
        "inconsistent_objects": 0.6,  # 不一致物体
        "missing_parts": 0.75,  # 缺失部分
    })
    # 物体数量异常阈值
    min_objects_for_scene: Dict[str, int] = field(default_factory=lambda: {
        "street": 2,
        "forest": 3,
        "indoor": 1,
    })


@dataclass
class CustomFormulaConfig:
    """自定义公式配置"""
    enabled: bool = False
    # 在这里定义你的自定义公式参数
    formula_alpha: float = 1.0
    formula_beta: float = 0.5
    formula_gamma: float = 0.3


@dataclass
class AIDetectorConfig:
    """AI 检测器完整配置"""
    feature_weights: FeatureWeights = field(default_factory=FeatureWeights)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    penalties: PenaltyParams = field(default_factory=PenaltyParams)
    boosts: BoostParams = field(default_factory=BoostParams)
    thresholds: ThresholdParams = field(default_factory=ThresholdParams)
    yolo_integration: YOLOIntegrationConfig = field(default_factory=YOLOIntegrationConfig)
    custom_formula: CustomFormulaConfig = field(default_factory=CustomFormulaConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "feature_weights": self.feature_weights.__dict__,
            "normalization": self.normalization.__dict__,
            "penalties": self.penalties.__dict__,
            "boosts": self.boosts.__dict__,
            "thresholds": self.thresholds.__dict__,
            "yolo_integration": {
                "enabled": self.yolo_integration.enabled,
                "object_detection_weight": self.yolo_integration.object_detection_weight,
                "abnormal_object_patterns": self.yolo_integration.abnormal_object_patterns,
                "min_objects_for_scene": self.yolo_integration.min_objects_for_scene,
            },
            "custom_formula": self.custom_formula.__dict__,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIDetectorConfig":
        """从字典加载配置"""
        config = cls()
        if "feature_weights" in data:
            config.feature_weights = FeatureWeights(**data["feature_weights"])
        if "normalization" in data:
            config.normalization = NormalizationParams(**data["normalization"])
        if "penalties" in data:
            config.penalties = PenaltyParams(**data["penalties"])
        if "boosts" in data:
            config.boosts = BoostParams(**data["boosts"])
        if "thresholds" in data:
            config.thresholds = ThresholdParams(**data["thresholds"])
        if "yolo_integration" in data:
            yolo_data = data["yolo_integration"]
            config.yolo_integration.enabled = yolo_data.get("enabled", False)
            config.yolo_integration.object_detection_weight = yolo_data.get("object_detection_weight", 0.20)
            config.yolo_integration.abnormal_object_patterns = yolo_data.get("abnormal_object_patterns", {})
            config.yolo_integration.min_objects_for_scene = yolo_data.get("min_objects_for_scene", {})
        if "custom_formula" in data:
            config.custom_formula = CustomFormulaConfig(**data["custom_formula"])
        return config


# 默认配置实例
DEFAULT_CONFIG = AIDetectorConfig()

