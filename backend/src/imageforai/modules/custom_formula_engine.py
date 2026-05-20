"""
自定义公式引擎
用于集成和测试你的自定义数学公式
"""

import math
import numpy as np
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field


@dataclass
class FormulaInput:
    """公式输入数据"""
    normalized_features: Dict[str, float]
    raw_features: Dict[str, float]
    context: Dict[str, Any]
    yolo_features: Optional[Dict[str, Any]] = None
    metadata_indicators: List[str] = field(default_factory=list)


@dataclass
class FormulaResult:
    """公式结果"""
    probability: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomFormula:
    """自定义公式基类"""

    def name(self) -> str:
        """公式名称"""
        return "base_formula"

    def description(self) -> str:
        """公式描述"""
        return "Base formula class"

    def calculate(self, inputs: FormulaInput) -> FormulaResult:
        """
        计算公式结果

        你需要重写这个方法！
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class WeightedSumFormula(CustomFormula):
    """
    加权求和公式（示例）
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "color_entropy": 0.2,
            "noise_level": 0.2,
            "edge_quality": 0.2,
            "texture_score": 0.2,
            "color_abnormality": 0.2,
        }

    def name(self) -> str:
        return "weighted_sum"

    def description(self) -> str:
        return "简单的加权求和公式"

    def calculate(self, inputs: FormulaInput) -> FormulaResult:
        score = 0.0
        for feature_name, weight in self.weights.items():
            value = inputs.normalized_features.get(feature_name, 0.0)
            score += value * weight

        # 简单的置信度计算
        active_features = sum(
            1 for v in inputs.normalized_features.values()
            if v > 0.3
        )
        confidence = min(active_features / len(self.weights), 1.0)

        return FormulaResult(
            probability=float(score),
            confidence=float(confidence),
            metadata={"formula_type": "weighted_sum"},
        )


class ExponentialFormula(CustomFormula):
    """
    指数公式（示例）
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def name(self) -> str:
        return "exponential"

    def description(self) -> str:
        return f"指数公式: p = 1 - exp(-α * sum(w_i * x_i) - β)"

    def calculate(self, inputs: FormulaInput) -> FormulaResult:
        """
        你的自定义公式写在这里！

        示例公式：
            p = 1 - exp(-α * sum(w_i * x_i) - β)

        这里的 α 和 β 是你可以调整的参数
        """
        # 计算特征总和
        feature_sum = 0.0
        feature_names = [
            "color_entropy",
            "noise_level",
            "edge_quality",
            "texture_score",
            "color_abnormality",
        ]

        for name in feature_names:
            feature_sum += inputs.normalized_features.get(name, 0.0)

        # 应用你的公式
        exponent = -self.alpha * feature_sum - self.beta
        probability = 1.0 - math.exp(exponent)

        # 裁剪到 [0, 1] 范围
        probability = max(min(probability, 1.0), 0.0)

        # 置信度计算
        confidence = min(feature_sum / len(feature_names), 1.0)

        return FormulaResult(
            probability=float(probability),
            confidence=float(confidence),
            metadata={
                "formula_type": "exponential",
                "alpha": self.alpha,
                "beta": self.beta,
                "feature_sum": feature_sum,
            },
        )


class CustomSigmoidFormula(CustomFormula):
    """
    自定义 Sigmoid 公式（更复杂的示例）
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def name(self) -> str:
        return "custom_sigmoid"

    def description(self) -> str:
        return (
            f"Sigmoid公式: p = 1 / (1 + exp(-α*(sum(wx) - β))) "
            f"调整参数: α={self.alpha}, β={self.beta}, γ={self.gamma}"
        )

    def calculate(self, inputs: FormulaInput) -> FormulaResult:
        """
        更复杂的自定义公式示例

        公式：
            z = α * (weighted_sum - β)
            p = 1 / (1 + exp(-z))

        你可以在这里修改公式！
        """
        # 基础特征权重
        base_weights = {
            "color_entropy": 0.15,
            "noise_level": 0.15,
            "edge_quality": 0.2,
            "texture_score": 0.2,
            "color_abnormality": 0.15,
            "blockiness": 0.15,
        }

        weighted_sum = 0.0
        for feature_name, weight in base_weights.items():
            value = inputs.normalized_features.get(feature_name, 0.0)
            weighted_sum += value * weight

        # 应用 Sigmoid 公式
        z = self.alpha * (weighted_sum - self.beta)

        # 防止溢出
        z = max(min(z, 20), -20)

        probability = 1.0 / (1.0 + math.exp(-z))

        # 结合 YOLO 特征（如果可用）
        if inputs.yolo_features is not None:
            yolo_abnormal = inputs.yolo_features.get("abnormal_score", 0.0)
            yolo_weight = self.gamma
            probability = (
                probability * (1 - yolo_weight) +
                yolo_abnormal * yolo_weight
            )

        # 置信度计算
        confidence = 0.0
        if inputs.metadata_indicators:
            confidence += 0.3

        active_features = sum(
            1 for v in inputs.normalized_features.values()
            if v > 0.3
        )
        confidence += min(active_features / len(base_weights) * 0.7, 0.7)
        confidence = min(confidence, 1.0)

        return FormulaResult(
            probability=float(probability),
            confidence=float(confidence),
            metadata={
                "formula_type": "custom_sigmoid",
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "weighted_sum": weighted_sum,
                "z": z,
            },
        )


# ========================================
# 在下面添加你的自定义公式！
# ========================================


class YourCustomFormula(CustomFormula):
    """
    这是你的自定义公式模板！

    使用方法：
        1. 重写 name() 和 description()
        2. 在 calculate() 中实现你的数学公式
        3. 在 FormulaRegistry 中注册它
    """

    def __init__(
        self,
        # 在这里添加你的公式参数
        param1: float = 1.0,
        param2: float = 0.5,
    ):
        self.param1 = param1
        self.param2 = param2

    def name(self) -> str:
        return "your_custom_formula"

    def description(self) -> str:
        return "在这里描述你的公式"

    def calculate(self, inputs: FormulaInput) -> FormulaResult:
        """
        在这里写你的数学公式！

        可用的输入：
            - inputs.normalized_features: 归一化特征（0-1之间）
            - inputs.raw_features: 原始特征值
            - inputs.context: 上下文信息
            - inputs.yolo_features: YOLO 特征（如果可用）
            - inputs.metadata_indicators: 元数据指示器
        """
        # ========================================
        # 在这里实现你的公式！
        # 例如：
        #   feature1 = inputs.normalized_features.get("color_entropy", 0.0)
        #   feature2 = inputs.normalized_features.get("noise_level", 0.0)
        #   probability = (feature1 * self.param1 + feature2) * self.param2
        # ========================================

        # 这是一个占位实现，替换成你的公式
        probability = 0.5
        confidence = 0.5

        return FormulaResult(
            probability=float(probability),
            confidence=float(confidence),
            metadata={
                "formula_type": "your_custom_formula",
                "param1": self.param1,
                "param2": self.param2,
            },
        )


class FormulaRegistry:
    """公式注册表"""

    def __init__(self):
        self._formulas: Dict[str, CustomFormula] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册默认公式"""
        self.register(WeightedSumFormula())
        self.register(ExponentialFormula())
        self.register(CustomSigmoidFormula())
        # 取消下面这行的注释来启用你的自定义公式
        # self.register(YourCustomFormula())

    def register(self, formula: CustomFormula):
        """注册一个公式"""
        self._formulas[formula.name()] = formula

    def get(self, name: str) -> Optional[CustomFormula]:
        """获取一个公式"""
        return self._formulas.get(name)

    def list_formulas(self) -> List[str]:
        """列出所有可用的公式"""
        return list(self._formulas.keys())

    def get_formula_info(self, name: str) -> Optional[Dict[str, str]]:
        """获取公式信息"""
        formula = self.get(name)
        if formula:
            return {
                "name": formula.name(),
                "description": formula.description(),
            }
        return None


# 全局注册表实例
registry = FormulaRegistry()


def apply_custom_formula(
    formula_name: str,
    inputs: FormulaInput,
) -> FormulaResult:
    """
    应用自定义公式

    参数:
        formula_name: 公式名称
        inputs: 公式输入数据

    返回:
        FormulaResult
    """
    formula = registry.get(formula_name)
    if formula is None:
        raise ValueError(f"Unknown formula: {formula_name}")

    return formula.calculate(inputs)

