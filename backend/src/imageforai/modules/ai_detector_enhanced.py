"""
增强版 AI 检测器
整合了配置管理、YOLO 特征融合和自定义公式
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import math
import cv2

from ..config import (
    AIDetectorConfig,
    DEFAULT_CONFIG,
)
from .yolo_feature_extractor import (
    YOLOFeatureExtractor,
    integrate_yolo_features,
)
from .custom_formula_engine import (
    FormulaInput,
    FormulaResult,
    registry,
    apply_custom_formula,
)


class EnhancedAIDetector:
    """增强版 AI 检测器"""

    def __init__(
        self,
        config: Optional[AIDetectorConfig] = None,
        custom_formula_name: Optional[str] = None,
    ):
        """
        初始化增强版检测器

        参数:
            config: 配置对象（如果不传则使用默认配置）
            custom_formula_name: 自定义公式名称（如果要使用自定义公式）
        """
        self.config = config or DEFAULT_CONFIG
        self.custom_formula_name = custom_formula_name

        # 初始化 YOLO 特征提取器（如果启用了）
        self.yolo_extractor: Optional[YOLOFeatureExtractor] = None
        if self.config.yolo_integration.enabled:
            self.yolo_extractor = YOLOFeatureExtractor()

    def detect(
        self,
        image_path: str,
        metadata_indicators: List[str] = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        检测图片是否为 AI 生成

        参数:
            image_path: 图片路径
            metadata_indicators: 元数据指示器列表
            context: 上下文信息

        返回:
            包含检测结果的字典
        """
        metadata_indicators = metadata_indicators or []
        context = context or {}

        try:
            # 1. 计算基础特征
            raw_features = self._calculate_raw_features(image_path)
            normalized_features = self._normalize_features(raw_features)

            # 2. 提取 YOLO 特征（如果启用）
            yolo_features = None
            if self.yolo_extractor is not None:
                yolo_features = self.yolo_extractor.extract_yolo_features(image_path)

            # 3. 构建上下文
            full_context = self._build_context(
                image_path,
                normalized_features,
                metadata_indicators,
                context,
            )

            # 4. 计算 AI 概率
            if self.custom_formula_name:
                # 使用自定义公式
                formula_input = FormulaInput(
                    normalized_features=normalized_features,
                    raw_features=raw_features,
                    context=full_context,
                    yolo_features=yolo_features,
                    metadata_indicators=metadata_indicators,
                )
                formula_result = apply_custom_formula(
                    self.custom_formula_name,
                    formula_input,
                )
                ai_probability = formula_result.probability
                confidence = formula_result.confidence
                formula_metadata = formula_result.metadata
            else:
                # 使用原始加权公式
                ai_probability = self._estimate_ai_probability(
                    normalized_features,
                    full_context,
                    yolo_features,
                )
                confidence = self._calculate_confidence(
                    normalized_features,
                    metadata_indicators,
                )
                formula_metadata = {}

            # 5. 确保概率在有效范围内
            ai_probability = max(min(ai_probability, 1.0), 0.0)
            is_likely_ai = ai_probability > self.config.thresholds.ai_probability_threshold

            return {
                "ai_probability": float(ai_probability),
                "confidence": float(confidence),
                "is_likely_ai": is_likely_ai,
                "features": raw_features,
                "normalized_features": normalized_features,
                "yolo_features": yolo_features,
                "context": full_context,
                "metadata_indicators": metadata_indicators,
                "formula_metadata": formula_metadata,
                "custom_formula_used": self.custom_formula_name,
            }

        except Exception as e:
            return {
                "error": str(e),
                "ai_probability": 0.0,
                "confidence": 0.0,
                "is_likely_ai": False,
            }

    def _calculate_raw_features(self, image_path: str) -> Dict[str, float]:
        """计算原始特征值"""
        features = {}

        # 颜色熵
        features["color_entropy"] = self._analyze_entropy(image_path)

        # 噪声水平
        features["noise_level"] = self._analyze_noise_pattern(image_path)

        # 块效应
        features["blockiness"] = self._analyze_blockiness(image_path)

        # 边缘质量
        features["edge_quality"] = self._analyze_edge_quality(image_path)

        # 颜色异常
        features["color_abnormality"] = self._analyze_color_abnormality(image_path)

        # 纹理分数
        features["texture_score"] = self._analyze_texture(image_path)

        return features

    def _normalize_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """归一化特征到 [0, 1] 范围"""
        norm = self.config.normalization
        normalized = {}

        # 颜色熵
        norm_val = (raw_features["color_entropy"] - norm.color_entropy_min) / (
            norm.color_entropy_max - norm.color_entropy_min
        )
        normalized["color_entropy"] = max(min(norm_val, 1.0), 0.0)

        # 噪声水平（对数归一化）
        normalized["noise_level"] = self._normalize_log(
            raw_features["noise_level"],
            norm.noise_level_min_log,
            norm.noise_level_max_log,
        )

        # 块效应
        normalized["blockiness"] = self._normalize_log(
            raw_features["blockiness"],
            norm.blockiness_min_log,
            norm.blockiness_max_log,
        )

        # 边缘质量
        normalized["edge_quality"] = self._normalize_log(
            raw_features["edge_quality"],
            norm.edge_quality_min_log,
            norm.edge_quality_max_log,
        )

        # 颜色异常
        normalized["color_abnormality"] = self._normalize_log(
            raw_features["color_abnormality"],
            norm.color_abnormality_min_log,
            norm.color_abnormality_max_log,
        )

        # 纹理分数
        normalized["texture_score"] = self._normalize_log(
            raw_features["texture_score"],
            norm.texture_score_min_log,
            norm.texture_score_max_log,
        )

        return normalized

    def _normalize_log(self, value: float, min_log: float, max_log: float) -> float:
        """对数归一化"""
        v = float(value or 0.0)
        if v < 0:
            v = 0.0
        if not math.isfinite(v):
            return 0.0
        log_v = float(math.log1p(v))
        if not math.isfinite(log_v):
            return 0.0
        if max_log <= min_log:
            return 0.0
        return float(min(max((log_v - min_log) / (max_log - min_log), 0.0), 1.0))

    def _build_context(
        self,
        image_path: str,
        normalized_features: Dict[str, float],
        metadata_indicators: List[str],
        input_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建完整的上下文信息"""
        context = dict(input_context)

        # 获取图像尺寸
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                context["width"] = width
                context["height"] = height
        except Exception:
            context["width"] = 0
            context["height"] = 0

        # 计算附加特征
        context["has_strong_ai_metadata"] = bool(metadata_indicators)
        context["has_camera_markers"] = False

        # 文档检测
        is_document_like = (
            normalized_features.get("edge_quality", 0.0) > 0.9
            and normalized_features.get("texture_score", 0.0) > 0.9
            and normalized_features.get("blockiness", 0.0) > 0.9
            and normalized_features.get("color_entropy", 0.0) == 0.0
            and normalized_features.get("color_abnormality", 0.0) == 0.0
        )
        context["is_document_like"] = is_document_like

        # 截图检测
        context["is_screenshot_like"] = self._is_screenshot_like(
            context.get("width", 0),
            context.get("height", 0),
            context.get("has_strong_ai_metadata", False),
            context.get("has_camera_markers", False),
        )

        return context

    def _estimate_ai_probability(
        self,
        normalized_features: Dict[str, float],
        context: Dict[str, Any],
        yolo_features: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        估计 AI 生成概率（使用配置中的参数）
        """
        weights = self.config.feature_weights
        penalties = self.config.penalties
        boosts = self.config.boosts

        # 加权求和
        weighted_sum = 0.0
        for feature_name, weight in weights.__dict__.items():
            if feature_name in normalized_features:
                value = normalized_features[feature_name]
                if math.isfinite(value):
                    weighted_sum += value * weight

        # 应用惩罚项
        penalty = 0.0

        # 文档惩罚
        if context.get("is_document_like"):
            penalty += penalties.document_penalty_base
        if context.get("color_diversity", 1.0) < 0.001:
            penalty += penalties.document_penalty_low_diversity
        if context.get("mean_entropy", 10) < 2.0:
            penalty += penalties.document_penalty_low_entropy
        if context.get("avg_correlation", 0) > 0.98:
            penalty += penalties.document_penalty_high_correlation

        # 相机惩罚
        if context.get("has_camera_markers") and not context.get("has_strong_ai_metadata"):
            penalty += penalties.camera_penalty_base
        if 0.08 <= context.get("noise_ratio", 0) <= 0.35:
            penalty += penalties.camera_penalty_noise_range

        # 截图惩罚
        if context.get("is_screenshot_like") and not context.get("has_strong_ai_metadata"):
            penalty += penalties.screenshot_penalty

        # 应用增强项
        boost = 0.0
        if context.get("has_strong_ai_metadata"):
            boost += boosts.metadata_boost_base
        boost += normalized_features.get("metadata_score", 0.0) * boosts.metadata_boost_score_weight

        # 基础概率
        base_probability = weighted_sum - penalty + boost

        # 融合 YOLO 特征
        if yolo_features is not None:
            base_probability = integrate_yolo_features(
                base_probability,
                yolo_features,
                self.config.yolo_integration.object_detection_weight,
            )

        return max(min(base_probability, 1.0), 0.0)

    def _calculate_confidence(
        self,
        normalized_features: Dict[str, float],
        metadata_indicators: List[str],
    ) -> float:
        """计算置信度"""
        confidence = 0.0

        # 元数据指示器加分
        if metadata_indicators:
            confidence += 0.3

        # 活跃特征数
        active_features = sum(
            1 for v in normalized_features.values()
            if v > 0.3
        )
        total_features = len(self.config.feature_weights.__dict__)
        if total_features > 0:
            confidence += min(active_features / total_features * 0.7, 0.7)

        return min(confidence, 1.0)

    # ========================================
    # 特征分析方法（保持原样）
    # ========================================

    def _analyze_entropy(self, image_path: str) -> float:
        """分析颜色熵"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        entropy_sum = 0
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
            hist = hist[hist > 0] / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            entropy_sum += entropy

        return float(entropy_sum / 3)

    def _analyze_noise_pattern(self, image_path: str) -> float:
        """分析噪声模式"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_var = np.var(laplacian)

        return float(noise_var)

    def _analyze_blockiness(self, image_path: str) -> float:
        """分析块效应"""
        block_size = 8
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        block_vars = []
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))

        return float(np.mean(block_vars)) if block_vars else 0.0

    def _analyze_edge_quality(self, image_path: str) -> float:
        """分析边缘质量"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_std = np.std(edge_magnitude)

        return float(edge_std / (edge_density + 0.01))

    def _analyze_color_abnormality(self, image_path: str) -> float:
        """分析颜色异常"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        r = img_array[:, :, 0].astype(np.int16)
        g = img_array[:, :, 1].astype(np.int16)
        b = img_array[:, :, 2].astype(np.int16)

        rg_diff = np.abs(r - g)
        rb_diff = np.abs(r - b)
        gb_diff = np.abs(g - b)

        avg_diff = (np.mean(rg_diff) + np.mean(rb_diff) + np.mean(gb_diff)) / 3

        return float(avg_diff)

    def _analyze_texture(self, image_path: str) -> float:
        """分析纹理"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)

        gabor_kernels = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            gabor_kernels.append(kernel)

        texture_responses = []
        for kernel in gabor_kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            texture_responses.append(np.std(filtered))

        return float(np.mean(texture_responses))

    def _is_screenshot_like(
        self,
        width: int,
        height: int,
        has_strong_ai_metadata: bool,
        has_camera_markers: bool,
    ) -> bool:
        """判断是否看起来像截图"""
        if has_strong_ai_metadata or has_camera_markers:
            return False

        if width <= 0 or height <= 0:
            return False

        aspect = max(width, height) / max(min(width, height), 1)
        if aspect < 1.55:
            return False

        common_widths = {720, 1080, 1125, 1170, 1242, 1284, 1440, 1536}
        if width in common_widths or height in common_widths:
            return True

        return max(width, height) <= 5000

