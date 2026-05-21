"""
预训练AI检测模型集成
无需重新训练，直接使用SOTA模型
"""

import os
from typing import Dict, Any, Optional
import sys


class PretrainedAIDetector:
    """
    预训练AI检测器包装器
    支持多种SOTA模型
    """
    
    def __init__(self, model_name: str = "huggingface"):
        self.model_name = model_name
        self._model = None
        self._available = False
        
        # 尝试初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            if self.model_name == "huggingface":
                self._init_huggingface()
            else:
                print(f"不支持的模型: {self.model_name}")
        except Exception as e:
            print(f"预训练模型初始化失败: {e}")
            print("将使用回退方案")
            self._available = False
    
    def _init_huggingface(self):
        """初始化HuggingFace模型"""
        try:
            from transformers import pipeline
            import torch
            
            # 检测GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {device}")
            
            # 使用轻量且效果好的模型
            self._model = pipeline(
                "image-classification",
                model="FalconAI/ImageGuardian",
                device=device
            )
            self._available = True
            print("预训练模型加载成功!")
            
        except ImportError:
            print("需要安装 transformers 和 torch")
            print("运行: pip install transformers torch pillow")
            self._available = False
        except Exception as e:
            print(f"模型加载失败: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._available
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        检测图片是否为AI生成
        
        Returns:
            {
                'ai_probability': 0.0-1.0,
                'confidence': 0.0-1.0,
                'raw_result': 原始结果
            }
        """
        if not self._available or not self._model:
            return {
                'ai_probability': 0.5,
                'confidence': 0.0,
                'error': '模型不可用'
            }
        
        try:
            # 预测
            results = self._model(image_path)
            
            # 解析结果
            ai_score = 0.0
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'ai' in label or 'generated' in label or 'fake' in label:
                    ai_score = score
                elif 'real' in label or 'natural' in label:
                    ai_score = 1 - score
            
            return {
                'ai_probability': float(ai_score),
                'confidence': float(results[0]['score']),
                'raw_result': results,
                'success': True
            }
            
        except Exception as e:
            return {
                'ai_probability': 0.5,
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }


class LightweightAIDetector:
    """
    轻量级AI检测器（无需大模型依赖）
    使用启发式规则 + 外部API（如果可用）
    """
    
    def __init__(self):
        pass
    
    def detect_with_apis(self, image_path: str) -> Dict[str, Any]:
        """
        尝试使用免费API检测（可选）
        
        注意: 需要网络连接和API密钥
        """
        # 这里可以集成免费API
        # 例如: Google Vision API (有免费额度)
        # 或者其他公开的检测API
        
        return {
            'ai_probability': 0.5,
            'confidence': 0.0,
            'note': 'API模式需要额外配置'
        }


def create_fusion_detector():
    """
    创建融合检测器（现有规则 + 预训练模型）
    """
    from .ai_detector import AIDetector
    
    class FusionDetector:
        def __init__(self):
            self.rule_based = AIDetector()
            self.pretrained = PretrainedAIDetector()
        
        def detect(self, image_path: str, metadata_indicators: list = None) -> Dict[str, Any]:
            # 1. 规则系统结果
            rule_result = self.rule_based.calculate_ai_score(
                image_path, 
                metadata_indicators or []
            )
            
            # 2. 预训练模型结果
            pretrained_result = {'ai_probability': 0.5}
            if self.pretrained.is_available():
                pretrained_result = self.pretrained.detect(image_path)
            
            # 3. 融合（加权平均）
            rule_weight = 0.6
            pretrained_weight = 0.4
            
            final_score = (
                rule_result['ai_probability'] * rule_weight +
                pretrained_result.get('ai_probability', 0.5) * pretrained_weight
            )
            
            return {
                'ai_probability': final_score,
                'confidence': rule_result['confidence'],
                'rule_based_result': rule_result,
                'pretrained_result': pretrained_result,
                'is_likely_ai': final_score > 0.65
            }
    
    return FusionDetector()
