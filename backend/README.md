# AI Image Detection Toolkit

一套用于检测AI生成图片的工具集，包含元数据提取、像素分析、AI检测、对象识别和隐写检测功能。

## 功能特性

- **元数据提取**：提取图片的EXIF信息和PNG Chunk数据
- **像素分析**：分析颜色分布、噪声特征、熵值、块效应等
- **AI生成检测**：基于多特征融合的AI生成图片检测算法
- **对象检测**：使用YOLOv8进行物体识别和场景分析
- **隐写检测**：检测图片中可能隐藏的信息

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 使用方法

### 命令行接口

```bash
# 运行全部分析
imageforai image.png --all

# 仅提取元数据
imageforai image.png --metadata

# 仅检测AI生成
imageforai image.png --ai

# 输出JSON格式
imageforai image.png --all --json
```

### Python API

```python
from imageforai import MetadataExtractor, AIDetector

# 提取元数据
extractor = MetadataExtractor()
metadata = extractor.extract_all_metadata('image.png')

# AI检测
detector = AIDetector()
result = detector.detect('image.png')
print(f"AI概率: {result['ai_probability']}")
```

## 模块说明

### MetadataExtractor
- `extract_exif()` - 提取EXIF信息
- `extract_png_chunks()` - 解析PNG Chunk
- `get_image_info()` - 获取图片基本信息
- `check_ai_generation_metadata()` - 检查AI生成元数据

### PixelAnalyzer
- `analyze_color_distribution()` - 颜色分布分析
- `analyze_noise()` - 噪声分析
- `analyze_entropy()` - 熵值计算
- `analyze_blockiness()` - 块效应检测

### AIDetector
- `detect()` - 检测图片是否为AI生成

### ObjectDetector
- `detect_objects()` - 检测对象
- `analyze_scene()` - 场景分析

### StegDetector
- `analyze_all()` - 隐写检测

## 技术栈

- Python 3.8+
- Pillow (图像处理)
- OpenCV (计算机视觉)
- NumPy/SciPy (数值计算)
- Ultralytics YOLOv8 (对象检测)
- exifread/piexif (元数据)

## 项目结构

```
src/
└── imageforai/
    ├── __init__.py
    ├── cli.py              # 命令行接口
    └── modules/
        ├── metadata_extractor.py  # 元数据提取
        ├── pixel_analyzer.py      # 像素分析
        ├── ai_detector.py         # AI检测
        ├── object_detector.py     # 对象检测
        └── steg_detector.py       # 隐写检测
```

## License

MIT License