import numpy as np
from PIL import Image
from typing import Dict, Any
import cv2

class StegDetector:
    def __init__(self):
        pass
    
    def analyze_lsb(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        lsb_signals = []
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            lsb_plane = channel_data & 1
            lsb_signals.append(lsb_plane.flatten())
        
        lsb_entropies = []
        for signal in lsb_signals:
            hist, _ = np.histogram(signal, bins=2)
            if hist[0] == 0 or hist[1] == 0:
                entropy = 0
            else:
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
            lsb_entropies.append(float(entropy))
        
        avg_lsb_entropy = float(np.mean(lsb_entropies))
        
        is_suspicious = avg_lsb_entropy > 0.9
        
        return {
            'lsb_entropies': {
                'red': lsb_entropies[0],
                'green': lsb_entropies[1],
                'blue': lsb_entropies[2],
                'average': avg_lsb_entropy,
            },
            'is_suspicious': is_suspicious,
            'confidence': float(min(avg_lsb_entropy * 1.1, 1.0)),
        }
    
    def analyze_lsb_parity(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        parity_scores = []
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            lsb_plane = channel_data & 1
            
            total_pixels = lsb_plane.size
            ones_count = np.sum(lsb_plane)
            zero_count = total_pixels - ones_count
            
            ratio = ones_count / total_pixels if total_pixels > 0 else 0.5
            deviation = abs(ratio - 0.5)
            parity_scores.append(float(deviation))
        
        avg_deviation = float(np.mean(parity_scores))
        
        is_suspicious = avg_deviation < 0.02
        
        return {
            'parity_deviations': {
                'red': parity_scores[0],
                'green': parity_scores[1],
                'blue': parity_scores[2],
                'average': avg_deviation,
            },
            'is_suspicious': is_suspicious,
            'confidence': float(min((0.5 - avg_deviation) * 2, 1.0)),
        }
    
    def analyze_file_size_anomaly(self, image_path: str) -> Dict[str, Any]:
        import os
        
        file_size = os.path.getsize(image_path)
        
        img = Image.open(image_path)
        width, height = img.size
        channels = len(img.getbands())
        
        expected_size = width * height * channels
        
        compression_ratio = expected_size / file_size if file_size > 0 else 0
        
        is_suspicious = compression_ratio < 1.5
        
        return {
            'file_size_bytes': file_size,
            'expected_raw_size_bytes': expected_size,
            'compression_ratio': float(compression_ratio),
            'is_suspicious': is_suspicious,
            'confidence': float(min(1 - compression_ratio / 10, 1.0)) if compression_ratio < 10 else 0.1,
        }
    
    def analyze_dct_coefficients(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        height, width = gray.shape
        block_size = 8
        
        high_freq_counts = []
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct = cv2.dct(block)
                high_freq = np.sum(np.abs(dct[1:, 1:]))
                high_freq_counts.append(high_freq)
        
        avg_high_freq = np.mean(high_freq_counts) if high_freq_counts else 0
        
        is_suspicious = avg_high_freq < 0.01
        
        return {
            'avg_high_freq_energy': float(avg_high_freq),
            'is_suspicious': is_suspicious,
            'confidence': float(min((0.05 - avg_high_freq) * 20, 1.0)) if avg_high_freq < 0.05 else 0.1,
        }
    
    def analyze_all(self, image_path: str) -> Dict[str, Any]:
        lsb_result = self.analyze_lsb(image_path)
        parity_result = self.analyze_lsb_parity(image_path)
        size_result = self.analyze_file_size_anomaly(image_path)
        dct_result = self.analyze_dct_coefficients(image_path)
        
        suspicious_count = sum([
            lsb_result['is_suspicious'],
            parity_result['is_suspicious'],
            size_result['is_suspicious'],
            dct_result['is_suspicious'],
        ])
        
        overall_confidence = (
            lsb_result['confidence'] +
            parity_result['confidence'] +
            size_result['confidence'] +
            dct_result['confidence']
        ) / 4
        
        return {
            'lsb_analysis': lsb_result,
            'parity_analysis': parity_result,
            'size_analysis': size_result,
            'dct_analysis': dct_result,
            'total_suspicious_tests': suspicious_count,
            'overall_confidence': float(overall_confidence),
            'is_likely_stego': suspicious_count >= 2,
        }