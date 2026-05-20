import os
import struct
import exifread
import piexif
from PIL import Image
from typing import Dict, Any, Optional

class MetadataExtractor:
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    def __init__(self):
        self.png_chunk_types = {
            'IHDR': 'Image header',
            'IDAT': 'Image data',
            'IEND': 'Image end',
            'PLTE': 'Palette',
            'tRNS': 'Transparency',
            'gAMA': 'Gamma',
            'cHRM': 'Chromaticity',
            'sRGB': 'Standard RGB',
            'iCCP': 'ICC profile',
            'tEXt': 'Textual data',
            'zTXt': 'Compressed text',
            'iTXt': 'International text',
            'bKGD': 'Background',
            'pHYs': 'Physical dimensions',
            'sBIT': 'Significant bits',
            'sPLT': 'Suggested palette',
            'hIST': 'Histogram',
            'tIME': 'Last modification time',
        }

    def _make_json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]
        if isinstance(value, bytes):
            return {
                'type': 'bytes',
                'length': len(value),
                'preview_hex': value[:32].hex(),
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    
    def extract_exif(self, image_path: str) -> Dict[str, Any]:
        exif_data = {}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                for tag, value in tags.items():
                    exif_data[str(tag)] = str(value)
        except Exception as e:
            exif_data['error'] = str(e)
        return exif_data
    
    def extract_png_chunks(self, image_path: str) -> Dict[str, Any]:
        chunks = {}
        try:
            with open(image_path, 'rb') as f:
                signature = f.read(8)
                if signature != self.PNG_SIGNATURE:
                    return {'error': 'Not a valid PNG file'}
                
                while True:
                    chunk_data = f.read(4)
                    if len(chunk_data) < 4:
                        break
                    
                    length = struct.unpack('>I', chunk_data)[0]
                    chunk_type = f.read(4).decode('ascii')
                    chunk_content = f.read(length)
                    crc = f.read(4)
                    
                    chunks[chunk_type] = {
                        'description': self.png_chunk_types.get(chunk_type, 'Unknown'),
                        'length': length,
                    }
                    
                    if chunk_type == 'tEXt':
                        parts = chunk_content.split(b'\x00', 1)
                        if len(parts) == 2:
                            chunks[chunk_type]['keyword'] = parts[0].decode('ascii', errors='replace')
                            chunks[chunk_type]['text'] = parts[1].decode('ascii', errors='replace')
                    
                    elif chunk_type == 'zTXt':
                        parts = chunk_content.split(b'\x00', 2)
                        if len(parts) >= 2:
                            chunks[chunk_type]['keyword'] = parts[0].decode('ascii', errors='replace')
                            chunks[chunk_type]['compression'] = parts[1][0] if parts[1] else None
                    
                    elif chunk_type == 'iTXt':
                        parts = chunk_content.split(b'\x00', 4)
                        if len(parts) >= 2:
                            chunks[chunk_type]['keyword'] = parts[0].decode('ascii', errors='replace')
                    
                    if chunk_type == 'IEND':
                        break
                        
        except Exception as e:
            chunks['error'] = str(e)
        
        return chunks
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        info = {}
        try:
            with Image.open(image_path) as img:
                info['format'] = img.format
                info['mode'] = img.mode
                info['size'] = list(img.size)
                info['width'], info['height'] = img.size
                info['bits_per_sample'] = img.bits
                info['channels'] = len(img.getbands())
                
                if hasattr(img, 'info'):
                    info['info'] = self._make_json_safe(img.info)
                    
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def extract_all_metadata(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {'error': 'File not found'}
        
        result = {
            'file_path': image_path,
            'file_size': os.path.getsize(image_path),
            'image_info': self.get_image_info(image_path),
            'exif': self.extract_exif(image_path),
        }
        
        if result['image_info'].get('format') == 'PNG':
            result['png_chunks'] = self.extract_png_chunks(image_path)
        
        return result
    
    def check_ai_generation_metadata(self, image_path: str) -> Dict[str, Any]:
        metadata = self.extract_all_metadata(image_path)
        ai_indicators = []
        
        exif = metadata.get('exif', {})
        png_chunks = metadata.get('png_chunks', {})
        
        for key in exif:
            lower_key = key.lower()
            if any(term in lower_key for term in ['ai', 'generated', 'dreamstudio', 'midjourney', 'stable diffusion', 'dalle']):
                ai_indicators.append(f"EXIF contains AI-related term: {key}")
        
        if 'tEXt' in png_chunks:
            text_data = png_chunks['tEXt']
            if isinstance(text_data, dict):
                keyword = text_data.get('keyword', '').lower()
                text = text_data.get('text', '').lower()
                if any(term in keyword for term in ['ai', 'generated', 'dreamstudio', 'midjourney']):
                    ai_indicators.append(f"PNG tEXt chunk contains AI keyword: {keyword}")
                if any(term in text for term in ['ai', 'generated', 'dreamstudio', 'midjourney', 'stable diffusion', 'dalle']):
                    ai_indicators.append(f"PNG tEXt chunk contains AI text: {text[:50]}...")
        
        return {
            'has_ai_metadata': len(ai_indicators) > 0,
            'indicators': ai_indicators,
            'metadata': metadata,
        }
