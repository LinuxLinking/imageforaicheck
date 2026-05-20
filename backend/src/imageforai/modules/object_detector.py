from ultralytics import YOLO
from typing import Dict, Any, List
import os

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