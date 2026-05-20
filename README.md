# AI Image Detection Toolkit

一套用于检测AI生成图片的完整工具集，包含元数据提取、像素分析、AI检测、对象识别和隐写检测功能。

## 功能特性

- **元数据提取**：提取图片的 EXIF 信息和 PNG Chunk 数据
- **像素分析**：分析颜色分布、噪声特征、熵值、块效应等
- **AI生成检测**：基于多特征融合的AI生成图片检测算法
- **对象检测**：使用 YOLOv8 进行物体识别和场景分析
- **隐写检测**：检测图片中可能隐藏的信息
- **Web界面**：提供友好的可视化界面和报告导出功能

## 技术栈

- **后端**：Python 3.8+, Flask
- **前端**：HTML5, JavaScript, Tailwind CSS
- **核心库**：Pillow, OpenCV, NumPy, SciPy, Ultralytics YOLOv8

## 项目结构

```
basemenufoecheck/
├── app.py                    # Flask 启动文件
├── uploads/                  # 上传文件临时目录
├── exports/                  # 导出报告目录
├── backend/                  # 后端代码
│   ├── src/imageforai/       # 核心模块
│   │   ├── cli.py           # 命令行接口
│   │   └── modules/         # 功能模块
│   ├── tests/               # 测试脚本
│   ├── requirements.txt
│   └── setup.py
└── frontend/                # 前端代码
    └── templates/index.html # Web界面
```

## 安装

```bash
# 进入项目目录
cd basemenufoecheck

# 安装依赖
pip install -r backend/requirements.txt
pip install flask flask-cors
```

## 使用方法

### 1. 启动 Web 服务

```bash
python app.py
```

服务启动后访问：http://127.0.0.1:5000

### 2. 命令行接口

```bash
# 安装包
pip install -e backend/

# 运行分析
imageforai images/test.png --all

# 仅提取元数据
imageforai images/test.png --metadata

# 仅检测AI生成
imageforai images/test.png --ai

# JSON 输出
imageforai images/test.png --all --json
```

### 3. Python API

```python
from imageforai import MetadataExtractor, AIDetector

# 提取元数据
extractor = MetadataExtractor()
metadata = extractor.extract_all_metadata('image.png')

# AI检测
detector = AIDetector()
result = detector.detect('image.png')
print(f"AI概率: {result['ai_probability']:.2%}")
```

## 模块说明

| 模块 | 功能 |
|------|------|
| `MetadataExtractor` | 提取 EXIF 和 PNG Chunk |
| `PixelAnalyzer` | 像素统计特征分析 |
| `AIDetector` | AI生成图片检测 |
| `ObjectDetector` | YOLOv8 对象检测 |
| `StegDetector` | 隐写检测 |

## 导出格式

支持三种报告导出格式：

- **Markdown** (.md) - 适合文档记录
- **TXT** (.txt) - 纯文本格式
- **JSON** (.json) - 结构化数据

## 开发

### 目录职责

| 目录 | 职责 |
|------|------|
| `backend/src/imageforai/modules/` | 核心算法模块 |
| `backend/tests/` | 单元测试 |
| `frontend/templates/` | Web界面 |
| `app.py` | API 入口 |

### 运行测试

```bash
cd backend
python -m pytest tests/test_detector.py -v
```

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！