import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from imageforai import MetadataExtractor, PixelAnalyzer, AIDetector, ObjectDetector, StegDetector

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'templates'))
CORS(app)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
EXPORT_FOLDER = os.path.join(BASE_DIR, 'exports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            extractor = MetadataExtractor()
            ai_detector = AIDetector()
            pixel_analyzer = PixelAnalyzer()
            steg_detector = StegDetector()
            
            metadata = extractor.extract_all_metadata(filepath)
            ai_result = ai_detector.detect(filepath)
            pixel_result = pixel_analyzer.analyze_all(filepath)
            steg_result = steg_detector.analyze_all(filepath)
            
            result = {
                'filename': file.filename,
                'metadata': metadata,
                'ai_detection': ai_result,
                'pixel_analysis': pixel_result,
                'steg_detection': steg_result,
            }
            
            os.remove(filepath)
            return jsonify(result)
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_result():
    data = request.json
    filename = data.get('filename', 'result')
    format_type = data.get('format', 'md')
    result = data.get('result', {})
    
    export_path = os.path.join(EXPORT_FOLDER, f"{filename}.{format_type}")
    
    if format_type == 'md':
        content = generate_markdown(result)
    elif format_type == 'txt':
        content = generate_text(result)
    elif format_type == 'json':
        content = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        return jsonify({'error': 'Unsupported format'}), 400
    
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return jsonify({'path': export_path, 'format': format_type})

@app.route('/api/export/download/<filename>')
def download_export(filename):
    filepath = os.path.join(EXPORT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

def generate_markdown(result):
    md = f"# AI Image Detection Report\n\n"
    md += f"## File: {result.get('filename', 'Unknown')}\n\n"
    
    if 'metadata' in result:
        meta = result['metadata']
        md += "## Metadata\n\n"
        md += f"- File Size: {meta.get('file_size', 'N/A')} bytes\n"
        img_info = meta.get('image_info', {})
        md += f"- Format: {img_info.get('format', 'N/A')}\n"
        md += f"- Dimensions: {img_info.get('width', 'N/A')}x{img_info.get('height', 'N/A')}\n"
        md += f"- Mode: {img_info.get('mode', 'N/A')}\n"
        
        if 'png_chunks' in meta:
            md += "\n### PNG Chunks\n\n"
            for chunk_type, info in meta['png_chunks'].items():
                if isinstance(info, dict):
                    md += f"- {chunk_type}: {info.get('description', 'Unknown')}\n"
    
    if 'ai_detection' in result:
        ai = result['ai_detection']
        md += "\n## AI Generation Detection\n\n"
        md += f"- AI Probability: {ai.get('ai_probability', 0):.2%}\n"
        md += f"- Confidence: {ai.get('confidence', 0):.2%}\n"
        md += f"- Likely AI Generated: {'Yes' if ai.get('is_likely_ai') else 'No'}\n"
        
        if ai.get('metadata_indicators'):
            md += "\n### Metadata Indicators\n\n"
            for indicator in ai['metadata_indicators']:
                md += f"- {indicator}\n"
    
    if 'pixel_analysis' in result:
        pixel = result['pixel_analysis']
        md += "\n## Pixel Analysis\n\n"
        md += f"- Color Diversity: {pixel.get('pixel_patterns', {}).get('color_diversity', 0):.4f}\n"
        md += f"- Mean Entropy: {pixel.get('entropy', {}).get('mean_entropy', 0):.2f}\n"
        md += f"- Noise Level: {'High' if pixel.get('noise_analysis', {}).get('is_high_noise') else 'Normal'}\n"
        md += f"- Blockiness: {'Detected' if pixel.get('blockiness', {}).get('is_blocky') else 'None'}\n"
    
    if 'steg_detection' in result:
        steg = result['steg_detection']
        md += "\n## Steganography Detection\n\n"
        md += f"- Suspicious Tests: {steg.get('total_suspicious_tests', 0)}/4\n"
        md += f"- Overall Confidence: {steg.get('overall_confidence', 0):.2%}\n"
        md += f"- Likely Contains Hidden Data: {'Yes' if steg.get('is_likely_stego') else 'No'}\n"
    
    md += "\n---\n*Generated by AI Image Detection Toolkit*\n"
    return md

def generate_text(result):
    text = "="*60 + "\n"
    text += "AI Image Detection Report\n"
    text += "="*60 + "\n\n"
    text += f"File: {result.get('filename', 'Unknown')}\n\n"
    
    if 'metadata' in result:
        meta = result['metadata']
        text += "[Metadata]\n"
        text += "-"*60 + "\n"
        text += f"  File Size: {meta.get('file_size', 'N/A')} bytes\n"
        img_info = meta.get('image_info', {})
        text += f"  Format: {img_info.get('format', 'N/A')}\n"
        text += f"  Dimensions: {img_info.get('width', 'N/A')}x{img_info.get('height', 'N/A')}\n"
        text += f"  Mode: {img_info.get('mode', 'N/A')}\n\n"
        
        if 'png_chunks' in meta:
            text += "  PNG Chunks:\n"
            for chunk_type, info in meta['png_chunks'].items():
                if isinstance(info, dict):
                    text += f"    {chunk_type}: {info.get('description', 'Unknown')}\n"
            text += "\n"
    
    if 'ai_detection' in result:
        ai = result['ai_detection']
        text += "[AI Generation Detection]\n"
        text += "-"*60 + "\n"
        text += f"  AI Probability: {ai.get('ai_probability', 0):.2%}\n"
        text += f"  Confidence: {ai.get('confidence', 0):.2%}\n"
        text += f"  Likely AI Generated: {'Yes' if ai.get('is_likely_ai') else 'No'}\n\n"
        
        if ai.get('metadata_indicators'):
            text += "  Metadata Indicators:\n"
            for indicator in ai['metadata_indicators']:
                text += f"    - {indicator}\n"
            text += "\n"
    
    if 'pixel_analysis' in result:
        pixel = result['pixel_analysis']
        text += "[Pixel Analysis]\n"
        text += "-"*60 + "\n"
        text += f"  Color Diversity: {pixel.get('pixel_patterns', {}).get('color_diversity', 0):.4f}\n"
        text += f"  Mean Entropy: {pixel.get('entropy', {}).get('mean_entropy', 0):.2f}\n"
        text += f"  Noise Level: {'High' if pixel.get('noise_analysis', {}).get('is_high_noise') else 'Normal'}\n"
        text += f"  Blockiness: {'Detected' if pixel.get('blockiness', {}).get('is_blocky') else 'None'}\n\n"
    
    if 'steg_detection' in result:
        steg = result['steg_detection']
        text += "[Steganography Detection]\n"
        text += "-"*60 + "\n"
        text += f"  Suspicious Tests: {steg.get('total_suspicious_tests', 0)}/4\n"
        text += f"  Overall Confidence: {steg.get('overall_confidence', 0):.2%}\n"
        text += f"  Likely Contains Hidden Data: {'Yes' if steg.get('is_likely_stego') else 'No'}\n\n"
    
    text += "="*60 + "\n"
    text += "Generated by AI Image Detection Toolkit\n"
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)