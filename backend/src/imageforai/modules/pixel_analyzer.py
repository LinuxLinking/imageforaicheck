import numpy as np
from PIL import Image
from scipy import stats
from typing import Dict, Any
import cv2

class PixelAnalyzer:
    def __init__(self):
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    
    def analyze_color_distribution(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        
        r_channel = img_array[:, :, 0].flatten()
        g_channel = img_array[:, :, 1].flatten()
        b_channel = img_array[:, :, 2].flatten()
        
        return {
            'red': {
                'mean': float(np.mean(r_channel)),
                'std': float(np.std(r_channel)),
                'min': int(np.min(r_channel)),
                'max': int(np.max(r_channel)),
                'skew': float(stats.skew(r_channel)),
                'kurtosis': float(stats.kurtosis(r_channel)),
            },
            'green': {
                'mean': float(np.mean(g_channel)),
                'std': float(np.std(g_channel)),
                'min': int(np.min(g_channel)),
                'max': int(np.max(g_channel)),
                'skew': float(stats.skew(g_channel)),
                'kurtosis': float(stats.kurtosis(g_channel)),
            },
            'blue': {
                'mean': float(np.mean(b_channel)),
                'std': float(np.std(b_channel)),
                'min': int(np.min(b_channel)),
                'max': int(np.max(b_channel)),
                'skew': float(stats.skew(b_channel)),
                'kurtosis': float(stats.kurtosis(b_channel)),
            },
        }
    
    def analyze_noise(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_std = float(np.std(edge_magnitude))
        
        noise_estimate = float(np.std(gray) / np.mean(gray))
        
        return {
            'laplacian_variance': laplacian_var,
            'edge_std': edge_std,
            'noise_ratio': noise_estimate,
            'is_high_noise': laplacian_var > 100,
        }
    
    def analyze_entropy(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        
        entropy_values = []
        for channel in range(3):
            channel_data = img_array[:, :, channel].flatten()
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
            hist = hist[hist > 0] / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            entropy_values.append(float(entropy))
        
        return {
            'red_entropy': entropy_values[0],
            'green_entropy': entropy_values[1],
            'blue_entropy': entropy_values[2],
            'mean_entropy': float(np.mean(entropy_values)),
        }
    
    def analyze_blockiness(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        block_size = 8
        height, width = gray.shape
        blockiness_score = 0
        blocks_count = 0
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_variance = np.var(block)
                blockiness_score += block_variance
                blocks_count += 1
        
        avg_blockiness = blockiness_score / blocks_count if blocks_count > 0 else 0
        
        return {
            'blockiness_score': float(avg_blockiness),
            'is_blocky': avg_blockiness < 50,
        }
    
    def analyze_pixel_patterns(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        
        pixel_values = img_array.reshape(-1, 3)
        unique_colors = len(np.unique(pixel_values, axis=0))
        total_pixels = pixel_values.shape[0]
        color_diversity = unique_colors / total_pixels
        
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        
        color_correlation = {
            'rg': float(np.corrcoef(r.flatten(), g.flatten())[0, 1]),
            'rb': float(np.corrcoef(r.flatten(), b.flatten())[0, 1]),
            'gb': float(np.corrcoef(g.flatten(), b.flatten())[0, 1]),
        }
        
        return {
            'unique_colors': unique_colors,
            'total_pixels': total_pixels,
            'color_diversity': float(color_diversity),
            'color_correlation': color_correlation,
        }
    
    def analyze_compression_artifacts(self, image_path: str) -> Dict[str, Any]:
        img_array = self.load_image(image_path)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        dct_blocks = []
        height, width = gray.shape
        block_size = 8
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct = cv2.dct(np.float32(block) / 255.0)
                dct_blocks.append(np.abs(dct))
        
        if dct_blocks:
            avg_dct = np.mean(dct_blocks, axis=0)
            high_freq_energy = np.sum(avg_dct[1:, 1:])
            low_freq_energy = np.sum(avg_dct[0, :] + avg_dct[:, 0] - avg_dct[0, 0])
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        else:
            freq_ratio = 0
        
        return {
            'high_low_freq_ratio': float(freq_ratio),
            'has_compression_artifacts': freq_ratio < 0.1,
        }
    
    def analyze_all(self, image_path: str) -> Dict[str, Any]:
        return {
            'color_distribution': self.analyze_color_distribution(image_path),
            'noise_analysis': self.analyze_noise(image_path),
            'entropy': self.analyze_entropy(image_path),
            'blockiness': self.analyze_blockiness(image_path),
            'pixel_patterns': self.analyze_pixel_patterns(image_path),
            'compression_artifacts': self.analyze_compression_artifacts(image_path),
        }