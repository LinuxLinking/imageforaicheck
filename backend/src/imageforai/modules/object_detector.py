from ultralytics import YOLO
from typing import Dict, Any, List
import os
import numpy as np

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8n.pt'):
        self.model = YOLO(model_name)
        self.model_path = model_name
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.5) -> Dict[str, Any]:
        try:
            results = self.model(image_path, conf=conf_threshold)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detection = {
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': {
                            'x1': float(box.xyxy[0][0]),
                            'y1': float(box.xyxy[0][1]),
                            'x2': float(box.xyxy[0][2]),
                            'y2': float(box.xyxy[0][3]),
                        },
                    }
                    detections.append(detection)
            
            return {
                'success': True,
                'detections': detections,
                'count': len(detections),
                'classes': list(set(d['class'] for d in detections)),
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'count': 0,
                'classes': [],
            }
    
    def detect_faces(self, image_path: str, conf_threshold: float = 0.5) -> Dict[str, Any]:
        try:
            results = self.model(image_path, conf=conf_threshold)
            
            faces = []
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == 'person':
                        faces.append({
                            'confidence': float(box.conf),
                            'bbox': {
                                'x1': float(box.xyxy[0][0]),
                                'y1': float(box.xyxy[0][1]),
                                'x2': float(box.xyxy[0][2]),
                                'y2': float(box.xyxy[0][3]),
                            },
                        })
            
            return {
                'success': True,
                'faces': faces,
                'count': len(faces),
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'faces': [],
                'count': 0,
            }
    
    def analyze_scene(self, image_path: str, conf_threshold: float = 0.5) -> Dict[str, Any]:
        result = self.detect_objects(image_path, conf_threshold)
        
        if not result['success']:
            return result
        
        class_counts = {}
        for detection in result['detections']:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        scene_description = []
        if 'person' in class_counts:
            scene_description.append(f"{class_counts['person']} person(s)")
        if 'car' in class_counts:
            scene_description.append(f"{class_counts['car']} car(s)")
        if 'dog' in class_counts:
            scene_description.append(f"{class_counts['dog']} dog(s)")
        if 'cat' in class_counts:
            scene_description.append(f"{class_counts['cat']} cat(s)")
        if 'tree' in class_counts:
            scene_description.append(f"{class_counts['tree']} tree(s)")
        if 'building' in class_counts:
            scene_description.append(f"{class_counts['building']} building(s)")
        
        return {
            'success': True,
            'detections': result['detections'],
            'class_counts': class_counts,
            'scene_description': ', '.join(scene_description) if scene_description else 'Unknown scene',
        }
    
    def extract_yolo_features(self, image_path: str) -> Dict[str, Any]:
        """
        提取 YOLOv8 相关的特征用于 AI 检测
        返回可以用于训练的数值特征
        """
        try:
            results = self.model(image_path, conf=0.1)
            
            detections = []
            confidences = []
            class_ids = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detections.append({
                            'class': result.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': [float(x) for x in box.xyxy[0]],
                        })
                        confidences.append(float(box.conf))
                        class_ids.append(int(box.cls))
            
            # 计算数值特征
            num_detections = len(detections)
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            max_confidence = float(np.max(confidences)) if confidences else 0.0
            min_confidence = float(np.min(confidences)) if confidences else 0.0
            std_confidence = float(np.std(confidences)) if confidences else 0.0
            
            # 类别多样性
            unique_classes = list(set(class_ids)) if class_ids else []
            num_unique_classes = len(unique_classes)
            
            # 检测到的类别统计（常见类别）
            common_classes = ['person', 'car', 'dog', 'cat', 'chair', 'table', 'cup', 'bottle']
            class_features = {}
            for cls in common_classes:
                class_features[f'has_{cls}'] = 1.0 if any(d['class'] == cls for d in detections) else 0.0
            
            # 综合异常分数
            # AI 生成图片通常检测置信度较低，或检测到奇怪的物体组合
            anomaly_score = 0.0
            if num_detections == 0:
                anomaly_score += 0.3  # 什么都没检测到
            elif avg_confidence < 0.3:
                anomaly_score += 0.4  # 整体置信度低
            elif std_confidence > 0.4:
                anomaly_score += 0.2  # 置信度差异大
            
            return {
                'success': True,
                'num_detections': num_detections,
                'num_unique_classes': num_unique_classes,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
                'std_confidence': std_confidence,
                'yolo_anomaly_score': min(anomaly_score, 1.0),
                'class_features': class_features,
                'detections': detections,
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'num_detections': 0,
                'num_unique_classes': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'std_confidence': 0.0,
                'yolo_anomaly_score': 0.0,
                'class_features': {},
                'detections': [],
            }