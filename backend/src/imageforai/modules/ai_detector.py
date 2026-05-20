import numpy as np
from PIL import Image
from typing import Dict, Any, List
import cv2
import math

class AIDetector:
    def __init__(self):
        self.features_weights = {
            'color_entropy': 0.15,
            'noise_level': 0.15,
            'blockiness': 0.15,
            'edge_quality': 0.15,
            'color_abnormality': 0.15,
            'texture_score': 0.15,
            'metadata_score': 0.10,
        }

    def _normalize_log(self, value: float, min_log: float, max_log: float) -> float:
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

    def _is_screenshot_like(self, width: int, height: int, has_strong_ai_metadata: bool, has_camera_markers: bool) -> bool:
        if has_strong_ai_metadata or has_camera_markers:
            return False
        w = int(width or 0)
        h = int(height or 0)
        if w <= 0 or h <= 0:
            return False
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < 1.55:
            return False
        common_widths = {720, 1080, 1125, 1170, 1242, 1284, 1440, 1536}
        if w in common_widths or h in common_widths:
            return True
        return max(w, h) <= 5000

    def estimate_ai_probability(self, normalized_features: Dict[str, float], context: Dict[str, Any]) -> float:
        weighted_sum = 0.0
        for key, weight in self.features_weights.items():
            value = float(normalized_features.get(key, 0.0) or 0.0)
            if not math.isfinite(value):
                value = 0.0
            weighted_sum += value * weight

        document_penalty = 0.0
        if context.get('is_document_like'):
            document_penalty += 0.28
        if float(context.get('color_diversity', 0.0) or 0.0) < 0.001:
            document_penalty += 0.08
        if float(context.get('mean_entropy', 0.0) or 0.0) < 2.0:
            document_penalty += 0.06
        if float(context.get('avg_correlation', 0.0) or 0.0) > 0.98:
            document_penalty += 0.05

        camera_penalty = 0.0
        if context.get('has_camera_markers') and not context.get('has_strong_ai_metadata'):
            camera_penalty += 0.12
        if 0.08 <= float(context.get('noise_ratio', 0.0) or 0.0) <= 0.35:
            camera_penalty += 0.03

        screenshot_penalty = 0.0
        if context.get("is_screenshot_like") and not context.get("has_strong_ai_metadata"):
            screenshot_penalty += 0.18

        metadata_boost = 0.0
        if context.get('has_strong_ai_metadata'):
            metadata_boost += 0.25
        metadata_boost += float(normalized_features.get('metadata_score', 0.0)) * 0.15

        probability = weighted_sum - document_penalty - camera_penalty - screenshot_penalty + metadata_boost
        if not math.isfinite(probability):
            probability = 0.0
        return float(min(max(probability, 0.0), 1.0))
    
    def analyze_edge_quality(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_std = np.std(edge_magnitude)
        
        return float(edge_std / (edge_density + 0.01))
    
    def analyze_color_abnormality(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        r = img_array[:, :, 0].astype(np.int16)
        g = img_array[:, :, 1].astype(np.int16)
        b = img_array[:, :, 2].astype(np.int16)
        
        rg_diff = np.abs(r - g)
        rb_diff = np.abs(r - b)
        gb_diff = np.abs(g - b)
        
        avg_diff = (np.mean(rg_diff) + np.mean(rb_diff) + np.mean(gb_diff)) / 3
        
        return float(avg_diff)
    
    def analyze_texture(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
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
    
    def analyze_noise_pattern(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_var = np.var(laplacian)
        
        return float(noise_var)
    
    def analyze_entropy(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        entropy_sum = 0
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
            hist = hist[hist > 0] / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            entropy_sum += entropy
        
        return float(entropy_sum / 3)
    
    def calculate_ai_score(self, image_path: str, metadata_indicators: List[str] = [], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        features = {}
        
        features['color_entropy'] = self.analyze_entropy(image_path)
        features['noise_level'] = self.analyze_noise_pattern(image_path)
        features['edge_quality'] = self.analyze_edge_quality(image_path)
        features['color_abnormality'] = self.analyze_color_abnormality(image_path)
        features['texture_score'] = self.analyze_texture(image_path)
        
        block_size = 8
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        block_vars = []
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))

        features['blockiness'] = float(np.mean(block_vars)) if block_vars else 0.0
        
        if metadata_indicators:
            features['metadata_score'] = 1.0
        else:
            features['metadata_score'] = 0.0
        
        normalized_features = {}
        normalized_features["color_entropy"] = float(min(max((float(features["color_entropy"]) - 2.0) / 6.0, 0.0), 1.0))
        normalized_features["noise_level"] = self._normalize_log(features["noise_level"], min_log=2.0, max_log=7.6)
        normalized_features["blockiness"] = self._normalize_log(features["blockiness"], min_log=3.2, max_log=7.6)
        normalized_features["edge_quality"] = self._normalize_log(features["edge_quality"], min_log=4.0, max_log=8.6)
        normalized_features["color_abnormality"] = self._normalize_log(features["color_abnormality"], min_log=2.4, max_log=5.6)
        normalized_features["texture_score"] = self._normalize_log(features["texture_score"], min_log=4.8, max_log=7.0)
        normalized_features["metadata_score"] = 1.0 if metadata_indicators else 0.0

        color_diversity = 0.0
        avg_correlation = 0.0
        try:
            pixel_values = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(pixel_values, axis=0))
            total_pixels = pixel_values.shape[0]
            color_diversity = unique_colors / total_pixels if total_pixels > 0 else 0.0
            r = img_array[:, :, 0].flatten()
            g = img_array[:, :, 1].flatten()
            b = img_array[:, :, 2].flatten()
            correlations = []
            for a, bch in [(r, g), (r, b), (g, b)]:
                if float(np.std(a)) < 1e-8 or float(np.std(bch)) < 1e-8:
                    continue
                corr = float(np.corrcoef(a, bch)[0, 1])
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations.append(corr)
            avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
        except Exception:
            avg_correlation = 0.0

        derived_context = {
            'is_document_like': normalized_features.get('edge_quality', 0.0) > 0.9 and normalized_features.get('texture_score', 0.0) > 0.9 and normalized_features.get('blockiness', 0.0) > 0.9 and normalized_features.get('color_entropy', 0.0) == 0.0 and normalized_features.get('color_abnormality', 0.0) == 0.0,
            'has_camera_markers': False,
            'has_strong_ai_metadata': bool(metadata_indicators),
            'color_diversity': color_diversity,
            'mean_entropy': features['color_entropy'],
            'avg_correlation': avg_correlation,
            'noise_ratio': float(np.std(gray) / np.mean(gray)) if float(np.mean(gray)) > 0 else 0.0,
            'is_screenshot_like': False,
        }
        merged_context = derived_context | (context or {})
        merged_context["is_screenshot_like"] = self._is_screenshot_like(
            int(merged_context.get("width", width) or 0),
            int(merged_context.get("height", height) or 0),
            bool(merged_context.get("has_strong_ai_metadata")),
            bool(merged_context.get("has_camera_markers")),
        )

        ai_probability = self.estimate_ai_probability(normalized_features, merged_context)
        if merged_context.get("has_camera_markers") and not merged_context.get("has_strong_ai_metadata"):
            ai_probability = float(min(ai_probability, 0.65))
        if not math.isfinite(ai_probability):
            ai_probability = 0.0
        
        confidence = 0.0
        active_features = sum(1 for v in normalized_features.values() if v > 0.3)
        confidence = min(active_features / len(self.features_weights), 1.0)
        
        return {
            'ai_probability': float(ai_probability),
            'confidence': float(confidence),
            'features': features,
            'normalized_features': normalized_features,
            'is_likely_ai': ai_probability > 0.7,
            'metadata_indicators': metadata_indicators,
        }
    
    def detect(self, image_path: str, metadata_indicators: List[str] = [], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            return self.calculate_ai_score(image_path, metadata_indicators, context)
        except Exception as e:
            return {
                'error': str(e),
                'ai_probability': 0.0,
                'confidence': 0.0,
            }
