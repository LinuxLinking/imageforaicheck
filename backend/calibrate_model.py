#!/usr/bin/env python3
"""
小样本模型校准工具
利用有限的AI样本优化现有规则系统
"""

import os
import sys
import json
from typing import Dict, List, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from imageforai.modules.ai_detector import AIDetector
from imageforai.modules.metadata_extractor import MetadataExtractor
from imageforai.modules.pixel_analyzer import PixelAnalyzer


class ModelCalibrator:
    def __init__(self):
        self.detector = AIDetector()
        self.metadata_extractor = MetadataExtractor()
        self.pixel_analyzer = PixelAnalyzer()
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """分析单张图片，提取所有特征"""
        print(f"分析: {image_path}")
        
        # 提取元数据
        metadata = self.metadata_extractor.extract_all_metadata(image_path)
        
        # 提取像素特征
        pixel = self.pixel_analyzer.analyze_all(image_path)
        
        # 提取AI检测特征
        metadata_indicators = self._extract_metadata_indicators(metadata)
        ai_result = self.detector.calculate_ai_score(image_path, metadata_indicators)
        
        return {
            'path': image_path,
            'metadata': metadata,
            'pixel': pixel,
            'ai_result': ai_result
        }
    
    def _extract_metadata_indicators(self, metadata: Dict) -> List[str]:
        """从元数据中提取AI指标"""
        indicators = []
        exif = metadata.get('exif', {})
        png_chunks = metadata.get('png_chunks', {})
        
        ai_terms = ['midjourney', 'stable diffusion', 'dalle', 'generated', 'ai']
        
        for key, value in exif.items():
            key_lower = str(key).lower()
            value_lower = str(value).lower()
            if any(term in value_lower for term in ai_terms):
                indicators.append(f"EXIF: {key}")
        
        if 'tEXt' in png_chunks:
            text = str(png_chunks['tEXt']).lower()
            if any(term in text for term in ai_terms):
                indicators.append("PNG: AI text")
        
        return indicators
    
    def analyze_dataset(self, ai_samples_dir: str, real_samples_dir: str) -> Dict:
        """分析整个数据集"""
        print("=" * 60)
        print("AI 图像检测 - 小样本校准工具")
        print("=" * 60)
        
        ai_results = []
        real_results = []
        
        # 分析AI样本
        print(f"\n[1/3] 分析AI生成样本...")
        if os.path.exists(ai_samples_dir):
            for img_path in Path(ai_samples_dir).glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    result = self.analyze_single_image(str(img_path))
                    ai_results.append(result)
        
        # 分析真实样本
        print(f"\n[2/3] 分析真实照片样本...")
        if os.path.exists(real_samples_dir):
            for img_path in Path(real_samples_dir).glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    result = self.analyze_single_image(str(img_path))
                    real_results.append(result)
        
        print(f"\n[3/3] 分析完成！")
        print(f"  - AI样本: {len(ai_results)} 张")
        print(f"  - 真实样本: {len(real_results)} 张")
        
        return {
            'ai_samples': ai_results,
            'real_samples': real_results
        }
    
    def generate_calibration_report(self, results: Dict) -> Dict:
        """生成校准报告"""
        ai_samples = results['ai_samples']
        real_samples = results['real_samples']
        
        print("\n" + "=" * 60)
        print("校准报告")
        print("=" * 60)
        
        # 统计AI样本得分
        ai_scores = [r['ai_result']['ai_probability'] for r in ai_samples]
        if ai_scores:
            print(f"\n【AI样本得分分布】")
            print(f"  平均: {sum(ai_scores)/len(ai_scores):.2%}")
            print(f"  最低: {min(ai_scores):.2%}")
            print(f"  最高: {max(ai_scores):.2%}")
        
        # 统计真实样本得分
        real_scores = [r['ai_result']['ai_probability'] for r in real_samples]
        if real_scores:
            print(f"\n【真实样本得分分布】")
            print(f"  平均: {sum(real_scores)/len(real_scores):.2%}")
            print(f"  最低: {min(real_scores):.2%}")
            print(f"  最高: {max(real_scores):.2%}")
        
        # 计算最佳阈值
        if ai_scores and real_scores:
            threshold, accuracy = self._find_best_threshold(ai_scores, real_scores)
            print(f"\n【推荐阈值】")
            print(f"  高风险阈值: {threshold:.2%}")
            print(f"  当前准确率: {accuracy:.1%}")
        
        # 分析特征重要性
        if ai_samples and real_samples:
            print(f"\n【特征分析】")
            feature_importance = self._analyze_feature_importance(ai_samples, real_samples)
            for feature, score in feature_importance.items():
                print(f"  {feature}: {score:.3f}")
        
        # 生成推荐配置
        config = self._generate_recommended_config(ai_scores, real_scores)
        
        return config
    
    def _find_best_threshold(self, ai_scores: List[float], real_scores: List[float]) -> Tuple[float, float]:
        """寻找最佳阈值"""
        all_scores = sorted(ai_scores + real_scores)
        best_acc = 0
        best_threshold = 0.5
        
        for threshold in [i/100 for i in range(5, 100, 5)]:
            tp = sum(1 for s in ai_scores if s >= threshold)
            tn = sum(1 for s in real_scores if s < threshold)
            acc = (tp + tn) / (len(ai_scores) + len(real_scores))
            
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return best_threshold, best_acc
    
    def _analyze_feature_importance(self, ai_samples: List[Dict], real_samples: List[Dict]) -> Dict[str, float]:
        """分析特征重要性"""
        feature_importance = {}
        
        # 比较AI和真实样本的特征差异
        first_ai = ai_samples[0]['ai_result']['normalized_features']
        features = list(first_ai.keys())
        
        for feature in features:
            ai_vals = [r['ai_result']['normalized_features'].get(feature, 0) for r in ai_samples]
            real_vals = [r['ai_result']['normalized_features'].get(feature, 0) for r in real_samples]
            
            ai_mean = sum(ai_vals) / len(ai_vals) if ai_vals else 0
            real_mean = sum(real_vals) / len(real_vals) if real_vals else 0
            
            # 差异越大越重要
            diff = abs(ai_mean - real_mean)
            feature_importance[feature] = diff
        
        # 排序
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _generate_recommended_config(self, ai_scores: List[float], real_scores: List[float]) -> Dict:
        """生成推荐配置"""
        config = {
            'thresholds': {
                'high_risk': 0.7,
                'medium_risk': 0.4
            },
            'weights': {},
            'notes': []
        }
        
        if ai_scores:
            avg_ai = sum(ai_scores) / len(ai_scores)
            
            # 如果AI样本得分偏低，提高相关特征权重
            if avg_ai < 0.6:
                config['notes'].append("AI样本得分偏低，建议提高YOLO和元数据权重")
                config['weights'] = {
                    'yolo_anomaly_score': 0.15,
                    'yolo_object_score': 0.12,
                    'metadata_score': 0.12
                }
        
        return config
    
    def save_results(self, results: Dict, output_path: str = "calibration_report.json"):
        """保存校准结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="小样本模型校准工具")
    parser.add_argument("--ai-dir", required=True, help="AI生成样本目录")
    parser.add_argument("--real-dir", required=True, help="真实照片样本目录")
    parser.add_argument("--output", default="calibration_report.json", help="输出报告路径")
    
    args = parser.parse_args()
    
    calibrator = ModelCalibrator()
    results = calibrator.analyze_dataset(args.ai_dir, args.real_dir)
    config = calibrator.generate_calibration_report(results)
    
    # 保存完整结果
    full_results = {
        'analysis': results,
        'recommended_config': config
    }
    calibrator.save_results(full_results, args.output)


if __name__ == '__main__':
    main()
