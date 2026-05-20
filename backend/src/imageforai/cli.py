import argparse
import json
import os
from typing import Dict

from .modules.metadata_extractor import MetadataExtractor
from .modules.pixel_analyzer import PixelAnalyzer
from .modules.ai_detector import AIDetector
from .modules.object_detector import ObjectDetector
from .modules.steg_detector import StegDetector

def print_json(data: Dict):
    print(json.dumps(data, indent=2, ensure_ascii=False))

def print_results(results: Dict):
    print("=" * 60)
    print("Image Analysis Report")
    print("=" * 60)
    
    if 'metadata' in results:
        meta = results['metadata']
        print("\n[Metadata]")
        print(f"  File: {meta.get('file_path', 'N/A')}")
        print(f"  Size: {meta.get('file_size', 'N/A')} bytes")
        img_info = meta.get('image_info', {})
        print(f"  Format: {img_info.get('format', 'N/A')}")
        print(f"  Dimensions: {img_info.get('width', 'N/A')}x{img_info.get('height', 'N/A')}")
        print(f"  Mode: {img_info.get('mode', 'N/A')}")
        
        if 'png_chunks' in meta:
            chunks = meta['png_chunks']
            print(f"\n  PNG Chunks ({len(chunks)}):")
            for chunk_type, info in chunks.items():
                if isinstance(info, dict):
                    print(f"    {chunk_type}: {info.get('description', 'Unknown')}")
    
    if 'ai_detection' in results:
        ai = results['ai_detection']
        print("\n[AI Generation Detection]")
        print(f"  AI Probability: {ai.get('ai_probability', 0):.2%}")
        print(f"  Confidence: {ai.get('confidence', 0):.2%}")
        print(f"  Likely AI Generated: {'Yes' if ai.get('is_likely_ai') else 'No'}")
        
        if ai.get('metadata_indicators'):
            print("\n  Metadata Indicators:")
            for indicator in ai['metadata_indicators']:
                print(f"    - {indicator}")
    
    if 'pixel_analysis' in results:
        pixel = results['pixel_analysis']
        print("\n[Pixel Analysis]")
        print(f"  Color Diversity: {pixel.get('pixel_patterns', {}).get('color_diversity', 0):.4f}")
        print(f"  Mean Entropy: {pixel.get('entropy', {}).get('mean_entropy', 0):.2f}")
        print(f"  Noise Level: {'High' if pixel.get('noise_analysis', {}).get('is_high_noise') else 'Normal'}")
        print(f"  Blockiness: {'Detected' if pixel.get('blockiness', {}).get('is_blocky') else 'None'}")
    
    if 'object_detection' in results:
        obj = results['object_detection']
        print("\n[Object Detection]")
        print(f"  Objects Detected: {obj.get('count', 0)}")
        if obj.get('classes'):
            print(f"  Classes: {', '.join(obj['classes'])}")
        if obj.get('scene_description'):
            print(f"  Scene: {obj['scene_description']}")
    
    if 'steg_detection' in results:
        steg = results['steg_detection']
        print("\n[Steganography Detection]")
        print(f"  Suspicious Tests: {steg.get('total_suspicious_tests', 0)}/4")
        print(f"  Overall Confidence: {steg.get('overall_confidence', 0):.2%}")
        print(f"  Likely Contains Hidden Data: {'Yes' if steg.get('is_likely_stego') else 'No'}")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='AI Image Detection Toolkit')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--metadata', action='store_true', help='Extract metadata')
    parser.add_argument('--pixel', action='store_true', help='Analyze pixel patterns')
    parser.add_argument('--ai', action='store_true', help='Detect AI generation')
    parser.add_argument('--object', action='store_true', help='Detect objects')
    parser.add_argument('--steg', action='store_true', help='Detect steganography')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: File not found - {args.image_path}")
        return
    
    if not args.all and not any([args.metadata, args.pixel, args.ai, args.object, args.steg]):
        args.all = True
    
    results = {}
    
    if args.metadata or args.all:
        extractor = MetadataExtractor()
        results['metadata'] = extractor.extract_all_metadata(args.image_path)
    
    if args.ai or args.all:
        ai_detector = AIDetector()
        meta_result = results.get('metadata', {})
        png_chunks = meta_result.get('png_chunks', {})
        exif = meta_result.get('exif', {})
        
        indicators = []
        for key in exif:
            if any(term in key.lower() for term in ['ai', 'generated', 'dreamstudio', 'midjourney', 'stable diffusion', 'dalle']):
                indicators.append(f"EXIF: {key}")
        
        if 'tEXt' in png_chunks and isinstance(png_chunks['tEXt'], dict):
            keyword = png_chunks['tEXt'].get('keyword', '').lower()
            text = png_chunks['tEXt'].get('text', '').lower()
            if any(term in keyword for term in ['ai', 'generated']):
                indicators.append(f"PNG tEXt keyword: {keyword}")
            if any(term in text for term in ['ai', 'generated', 'dreamstudio', 'midjourney', 'stable diffusion', 'dalle']):
                indicators.append(f"PNG tEXt contains AI text")
        
        results['ai_detection'] = ai_detector.detect(args.image_path, indicators)
    
    if args.pixel or args.all:
        analyzer = PixelAnalyzer()
        results['pixel_analysis'] = analyzer.analyze_all(args.image_path)
    
    if args.object or args.all:
        detector = ObjectDetector()
        results['object_detection'] = detector.analyze_scene(args.image_path)
    
    if args.steg or args.all:
        steg_detector = StegDetector()
        results['steg_detection'] = steg_detector.analyze_all(args.image_path)
    
    if args.json:
        print_json(results)
    else:
        print_results(results)

if __name__ == '__main__':
    main()