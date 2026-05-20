import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from imageforai import MetadataExtractor, PixelAnalyzer, AIDetector, ObjectDetector, StegDetector

def get_test_images():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    if os.path.exists(data_dir):
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return []

def test_metadata_extractor():
    print("Testing Metadata Extractor...")
    extractor = MetadataExtractor()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            metadata = extractor.extract_all_metadata(img_path)
            print(f"  Format: {metadata.get('image_info', {}).get('format')}")
            print(f"  Size: {metadata.get('file_size')} bytes")
            print(f"  Dimensions: {metadata.get('image_info', {}).get('size')}")
            
            if 'png_chunks' in metadata:
                print(f"  PNG Chunks: {len(metadata['png_chunks'])}")
    else:
        print("  No test images found")
    
    print("✓ Metadata Extractor test completed")

def test_pixel_analyzer():
    print("\nTesting Pixel Analyzer...")
    analyzer = PixelAnalyzer()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = analyzer.analyze_all(img_path)
            print(f"  Mean Entropy: {results['entropy']['mean_entropy']:.2f}")
            print(f"  Noise Level: {'High' if results['noise_analysis']['is_high_noise'] else 'Normal'}")
            print(f"  Blockiness: {'Detected' if results['blockiness']['is_blocky'] else 'None'}")
    else:
        print("  No test images found")
    
    print("✓ Pixel Analyzer test completed")

def test_ai_detector():
    print("\nTesting AI Detector...")
    detector = AIDetector()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = detector.detect(img_path)
            print(f"  AI Probability: {results['ai_probability']:.2%}")
            print(f"  Confidence: {results['confidence']:.2%}")
            print(f"  Likely AI Generated: {'Yes' if results['is_likely_ai'] else 'No'}")
    else:
        print("  No test images found")
    
    print("✓ AI Detector test completed")

def test_steg_detector():
    print("\nTesting Steganography Detector...")
    detector = StegDetector()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = detector.analyze_all(img_path)
            print(f"  Suspicious Tests: {results['total_suspicious_tests']}/4")
            print(f"  Overall Confidence: {results['overall_confidence']:.2%}")
            print(f"  Likely Stego: {'Yes' if results['is_likely_stego'] else 'No'}")
    else:
        print("  No test images found")
    
    print("✓ Steganography Detector test completed")

if __name__ == '__main__':
    test_metadata_extractor()
    test_pixel_analyzer()
    test_ai_detector()
    test_steg_detector()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)