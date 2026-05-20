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

    def estimate_ai_probability(self, normalized_features: Dict[str, float], context: Dict[str, Any]) -> float:
        weighted_sum = 0.0
        for key, weight in self.features_weights.items():
            weighted_sum += float(normalized_features.get(key, 0.0)) * weight

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

        metadata_boost = 0.0
        if context.get('has_strong_ai_metadata'):
            metadata_boost += 0.25
        metadata_boost += float(normalized_features.get('metadata_score', 0.0)) * 0.15

        probability = weighted_sum - document_penalty - camera_penalty + metadata_boost
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
        
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
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
        
        features['blockiness'] = float(np.mean(block_vars))
        
        if metadata_indicators:
            features['metadata_score'] = 1.0
        else:
            features['metadata_score'] = 0.0
        
        normalized_features = {}
        thresholds = {
            'color_entropy': (5.0, 7.0),
            'noise_level': (50.0, 200.0),
            'blockiness': (30.0, 100.0),
            'edge_quality': (10.0, 50.0),
            'color_abnormality': (10.0, 50.0),
            'texture_score': (10.0, 50.0),
            'metadata_score': (0.0, 1.0),
        }
        
        for key, value in features.items():
            min_val, max_val = thresholds[key]
            normalized_features[key] = min(max((value - min_val) / (max_val - min_val), 0), 1)

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
            correlations = [
                float(np.corrcoef(r, g)[0, 1]),
                float(np.corrcoef(r, b)[0, 1]),
                float(np.corrcoef(g, b)[0, 1]),
            ]
            avg_correlation = sum(correlations) / len(correlations)
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
        }
        merged_context = derived_context | (context or {})

        ai_probability = self.estimate_ai_probability(normalized_features, merged_context)
        
        confidence = 0.0
        active_features = sum(1 for v in normalized_features.values() if v > 0.3)
        confidence = min(active_features / len(self.features_weights), 1.0)
        
        return {
            'ai_probability': float(ai_probability),
            'confidence': float(confidence),
            'features': features,
            'normalized_features': normalized_features,
            'is_likely_ai': ai_probability > 0.6,
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
