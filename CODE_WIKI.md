# AI Image Detection Toolkit - Code Wiki

## 1. 项目概述

### 1.1 项目简介
AI Image Detection Toolkit 是一套用于检测 AI 生成图片的完整工具集，提供元数据提取、像素分析、AI 检测、对象识别和隐写检测功能。

### 1.2 技术栈
- **后端框架**: Python 3.8+, Flask
- **前端**: HTML5, JavaScript
- **核心库**:
  - Pillow: 图像处理
  - OpenCV: 计算机视觉
  - NumPy/SciPy: 数值计算
  - Ultralytics YOLOv8: 对象检测
  - exifread/piexif: 元数据提取

### 1.3 目录结构
```
/workspace/
├── app.py                              # Flask 启动文件
├── backend/
│   ├── src/
│   │   └── imageforai/
│   │       ├── __init__.py
│   │       ├── cli.py                 # 命令行接口
│   │       ├── api/                   # API 路由层
│   │       │   ├── __init__.py
│   │       │   ├── routes/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── analyze.py     # 分析 API
│   │       │   │   └── export.py      # 导出 API
│   │       │   └── schemas/
│   │       │       ├── __init__.py
│   │       │       ├── error_codes.py
│   │       │       ├── request.py
│   │       │       └── response.py
│   │       ├── domain/                # 领域模型
│   │       │   ├── __init__.py
│   │       │   └── models.py
│   │       ├── infra/                 # 基础设施层
│   │       │   ├── __init__.py
│   │       │   └── storage/
│   │       │       ├── __init__.py
│   │       │       ├── history_repository.py
│   │       │       └── task_repository.py
│   │       ├── modules/               # 核心功能模块
│   │       │   ├── __init__.py
│   │       │   ├── metadata_extractor.py  # 元数据提取
│   │       │   ├── pixel_analyzer.py      # 像素分析
│   │       │   ├── ai_detector.py         # AI 检测
│   │       │   ├── object_detector.py     # 对象检测
│   │       │   └── steg_detector.py       # 隐写检测
│   │       ├── services/              # 服务层
│   │       │   ├── __init__.py
│   │       │   ├── analysis_service.py   # 分析服务
│   │       │   └── export_service.py     # 导出服务
│   │       └── web/                   # Web 应用
│   │           ├── __init__.py
│   │           └── app_factory.py
│   ├── tests/
│   │   └── test_detector.py
│   ├── requirements.txt
│   ├── setup.py
│   └── README.md
├── frontend/
│   └── templates/
│       └── index.html
├── exports/
└── README.md
```

---

## 2. 核心架构设计

### 2.1 架构分层
项目采用经典的分层架构设计：

1. **API 层** (`api/`)：处理 HTTP 请求，参数校验
2. **服务层** (`services/`)：业务逻辑编排
3. **领域层** (`domain/`)：数据模型定义
4. **基础设施层** (`infra/`)：存储、外部服务集成
5. **核心模块** (`modules/`)：核心功能实现

### 2.2 核心流程
```
用户请求 → API 路由 → 服务层 → 核心模块 → 数据模型 → 存储 → 返回响应
```

---

## 3. 核心模块详解

### 3.1 MetadataExtractor - 元数据提取模块
**文件路径**: `/workspace/backend/src/imageforai/modules/metadata_extractor.py`

**主要功能**:
- 提取图片 EXIF 信息
- 解析 PNG Chunk 数据
- 获取图片基本信息
- 检测 AI 生成相关元数据

**核心方法**:

| 方法名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `extract_exif()` | 提取 EXIF 数据 | `image_path: str` | `Dict[str, Any]` |
| `extract_png_chunks()` | 解析 PNG 块 | `image_path: str` | `Dict[str, Any]` |
| `get_image_info()` | 获取图片信息 | `image_path: str` | `Dict[str, Any]` |
| `extract_all_metadata()` | 提取全部元数据 | `image_path: str` | `Dict[str, Any]` |
| `check_ai_generation_metadata()` | 检测 AI 生成元数据 | `image_path: str` | `Dict[str, Any]` |

**使用示例**:
```python
from imageforai.modules.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract_all_metadata('image.png')
print(metadata['image_info'])
```

---

### 3.2 AIDetector - AI 生成检测模块
**文件路径**: `/workspace/backend/src/imageforai/modules/ai_detector.py`

**主要功能**:
- 基于多特征融合的 AI 图片检测
- 分析图片的视觉特征（熵、噪声、边缘等）
- 结合元数据信号进行综合评估

**核心特征权重**:
```python
features_weights = {
    'color_entropy': 0.15,      # 颜色熵
    'noise_level': 0.15,        # 噪声水平
    'blockiness': 0.15,         # 块效应
    'edge_quality': 0.15,       # 边缘质量
    'color_abnormality': 0.15,  # 颜色异常
    'texture_score': 0.15,      # 纹理分数
    'metadata_score': 0.10,     # 元数据分数
}
```

**核心方法**:

| 方法名 | 描述 |
|--------|------|
| `estimate_ai_probability()` | 估算 AI 生成概率 |
| `analyze_entropy()` | 分析颜色熵 |
| `analyze_noise_pattern()` | 分析噪声模式 |
| `analyze_edge_quality()` | 分析边缘质量 |
| `analyze_texture()` | 分析纹理特征 |
| `detect()` | 主检测方法 |

**使用示例**:
```python
from imageforai.modules.ai_detector import AIDetector

detector = AIDetector()
result = detector.detect('image.png', metadata_indicators=[], context={})
print(f"AI 概率: {result['ai_probability']}")
```

---

### 3.3 PixelAnalyzer - 像素分析模块
**文件路径**: `/workspace/backend/src/imageforai/modules/pixel_analyzer.py`

**主要功能**:
- 颜色分布统计
- 噪声特征分析
- 熵值计算
- 块效应检测
- 像素模式分析
- 压缩伪影检测

**核心方法**:

| 方法名 | 描述 |
|--------|------|
| `analyze_color_distribution()` | 颜色分布分析 |
| `analyze_noise()` | 噪声分析 |
| `analyze_entropy()` | 熵值计算 |
| `analyze_blockiness()` | 块效应分析 |
| `analyze_pixel_patterns()` | 像素模式分析 |
| `analyze_all()` | 执行全部分析 |

**使用示例**:
```python
from imageforai.modules.pixel_analyzer import PixelAnalyzer

analyzer = PixelAnalyzer()
result = analyzer.analyze_all('image.png')
print(f"平均熵值: {result['entropy']['mean_entropy']}")
```

---

### 3.4 StegDetector - 隐写检测模块
**文件路径**: `/workspace/backend/src/imageforai/modules/steg_detector.py`

**主要功能**:
- LSB（最低有效位）分析
- 奇偶性分析
- 文件大小异常检测
- DCT 系数分析

**核心检测方法**:

| 方法名 | 描述 |
|--------|------|
| `analyze_lsb()` | LSB 平面分析 |
| `analyze_lsb_parity()` | LSB 奇偶性分析 |
| `analyze_file_size_anomaly()` | 文件大小异常检测 |
| `analyze_dct_coefficients()` | DCT 系数分析 |
| `analyze_all()` | 执行全部分析 |

**使用示例**:
```python
from imageforai.modules.steg_detector import StegDetector

detector = StegDetector()
result = detector.analyze_all('image.png')
print(f"可疑测试数量: {result['total_suspicious_tests']}")
```

---

### 3.5 ObjectDetector - 对象检测模块
**文件路径**: `/workspace/backend/src/imageforai/modules/object_detector.py`

**主要功能**:
- 使用 YOLOv8 进行对象检测
- 人脸检测
- 场景分析

**核心方法**:

| 方法名 | 描述 |
|--------|------|
| `detect_objects()` | 检测对象 |
| `detect_faces()` | 人脸检测 |
| `analyze_scene()` | 场景分析 |

**使用示例**:
```python
from imageforai.modules.object_detector import ObjectDetector

detector = ObjectDetector('yolov8n.pt')
result = detector.detect_objects('image.png')
print(f"检测到 {result['count']} 个对象")
```

---

## 4. 服务层详解

### 4.1 AnalysisService - 分析服务
**文件路径**: `/workspace/backend/src/imageforai/services/analysis_service.py`

**主要职责**:
- 编排各分析模块
- 综合评估风险
- 生成检测报告

**支持的分析模式**:
```python
SUPPORTED_MODES = {
    'all',      # 全部分析
    'ai',       # 仅 AI 检测
    'metadata', # 仅元数据
    'pixel',    # 仅像素分析
    'steg',     # 仅隐写检测
}
```

**AI 元数据关键词** (用于检测元数据中的 AI 生成标识):
```python
AI_METADATA_VALUE_TERMS = (
    'midjourney', 'stable diffusion', 'stability ai', 'stabilityai',
    'automatic1111', 'comfyui', 'fooocus', 'invokeai', 'dreamstudio',
    'dall-e', 'dalle', 'firefly', 'flux', 'sdxl', 'novelai',
    'generative fill', 'generated by ai', 'ai generated',
)
```

**相机正面标识关键词**:
```python
CAMERA_POSITIVE_TERMS = (
    'canon', 'nikon', 'sony', 'fujifilm', 'panasonic', 'leica', 'hasselblad',
    'iphone', 'huawei', 'xiaomi', 'oppo', 'vivo',
    'directly photographed', 'digital camera', 'lens',
    'lightroom', 'camera raw',
)
```

**风险评估逻辑**:
1. 元数据信号权重：50%
2. 视觉信号权重：40%
3. 水印/隐写信号权重：10%
4. 根据检测结果计算综合风险分

**风险等级判定**:
- `high`: 风险分 ≥ 75 或存在强 AI 元数据
- `medium`: 45 ≤ 风险分 < 75
- `low`: 0 < 风险分 < 45
- `unknown`: 风险分 = 0

---

### 4.2 ExportService - 导出服务
**文件路径**: `/workspace/backend/src/imageforai/services/export_service.py`

**主要职责**:
- 导出检测报告
- 支持多种格式

**支持的导出格式**:
1. **Markdown** (`.md`)
2. **纯文本** (`.txt`)
3. **JSON** (`.json`)

**核心方法**:

| 方法名 | 描述 |
|--------|------|
| `export()` | 导出报告 |
| `generate_markdown()` | 生成 Markdown 格式 |
| `generate_text()` | 生成纯文本格式 |

---

## 5. API 层详解

### 5.1 分析 API
**文件路径**: `/workspace/backend/src/imageforai/api/routes/analyze.py`

**路由**:
- `POST /api/analyze?mode={mode}`: 分析图片

**请求参数**:
- `mode`: 分析模式（可选，默认 `all`）
- `file`: 上传的图片文件（multipart/form-data）

**响应示例**:
```json
{
  "success": true,
  "data": {
    "filename": "image.png",
    "mode": "all",
    "risk": {
      "level": "high",
      "ai_probability": 0.85,
      "confidence": 0.9,
      "score": 85.5,
      "evidence_summary": [...],
      "explanation": {...}
    },
    "evidence": {
      "ai_detection": {...},
      "metadata": {...},
      "pixel_analysis": {...},
      "steg_detection": {...}
    }
  }
}
```

---

### 5.2 导出 API
**文件路径**: `/workspace/backend/src/imageforai/api/routes/export.py`

**路由**:
1. `POST /api/export`: 导出报告
2. `GET /api/export/download/{filename}`: 下载导出的文件
3. `GET /api/history?page={page}&page_size={page_size}`: 获取历史记录
4. `GET /api/tasks/{task_id}`: 获取任务详情

**导出请求示例**:
```json
{
  "filename": "report",
  "format": "md",
  "result": {...}
}
```

---

## 6. 领域模型
**文件路径**: `/workspace/backend/src/imageforai/domain/models.py`

### 6.1 RiskSummary
风险摘要数据类。

**字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `level` | `str` | 风险等级 (high/medium/low/unknown) |
| `ai_probability` | `float` | AI 概率 (0-1) |
| `confidence` | `float` | 置信度 (0-1) |
| `summary` | `str` | 摘要文本 |
| `score` | `float` | 风险分数 (0-100) |
| `evidence_summary` | `List[str]` | 证据摘要列表 |
| `explanation` | `Dict[str, Any]` | 详细解释 |

---

### 6.2 EvidenceBundle
证据包数据类。

**字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `ai_detection` | `Optional[Dict]` | AI 检测结果 |
| `metadata` | `Optional[Dict]` | 元数据 |
| `pixel_analysis` | `Optional[Dict]` | 像素分析结果 |
| `steg_detection` | `Optional[Dict]` | 隐写检测结果 |

---

### 6.3 DetectionResult
检测结果数据类。

**字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `filename` | `str` | 文件名 |
| `mode` | `str` | 分析模式 |
| `risk` | `RiskSummary` | 风险摘要 |
| `task_id` | `str` | 任务 ID |
| `created_at` | `str` | 创建时间 |
| `evidence` | `EvidenceBundle` | 证据包 |

---

### 6.4 HistoryItem
历史记录项数据类。

**字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `task_id` | `str` | 任务 ID |
| `filename` | `str` | 文件名 |
| `mode` | `str` | 分析模式 |
| `created_at` | `str` | 创建时间 |
| `risk` | `RiskSummary` | 风险摘要 |

---

## 7. 基础设施层

### 7.1 JsonHistoryRepository - 历史记录存储
**文件路径**: `/workspace/backend/src/imageforai/infra/storage/history_repository.py`

**功能**:
- JSON 文件存储历史记录
- 分页查询

---

### 7.2 JsonTaskRepository - 任务存储
**文件路径**: `/workspace/backend/src/imageforai/infra/storage/task_repository.py`

**功能**:
- JSON 文件存储任务详情
- 按任务 ID 查询

---

## 8. Web 应用工厂
**文件路径**: `/workspace/backend/src/imageforai/web/app_factory.py`

**功能**:
- 创建 Flask 应用
- 配置路由和中间件
- 注册蓝图
- 初始化存储和服务

---

## 9. 运行和部署

### 9.1 安装依赖
```bash
cd /workspace
pip install -r backend/requirements.txt
pip install flask flask-cors
```

### 9.2 启动服务
```bash
python app.py
```
服务将在 `http://localhost:5000` 启动。

### 9.3 命令行使用
```bash
# 安装包
cd /workspace/backend
pip install -e .

# 运行全部分析
imageforai image.png --all

# 仅提取元数据
imageforai image.png --metadata

# 仅检测 AI 生成
imageforai image.png --ai

# JSON 输出
imageforai image.png --all --json
```

---

## 10. 测试
**文件路径**: `/workspace/backend/tests/test_detector.py`

运行测试:
```bash
cd /workspace/backend
python -m pytest tests/test_detector.py -v
```

---

## 11. 依赖关系
**文件路径**: `/workspace/backend/requirements.txt`

| 依赖 | 版本 | 用途 |
|------|------|------|
| Pillow | ≥ 9.0.0 | 图像处理 |
| numpy | ≥ 1.21.0 | 数值计算 |
| scipy | ≥ 1.7.0 | 科学计算 |
| opencv-python | ≥ 4.5.0 | 计算机视觉 |
| ultralytics | ≥ 8.0.0 | YOLO 对象检测 |
| exifread | ≥ 2.3.0 | EXIF 提取 |
| piexif | ≥ 1.1.3 | EXIF 处理 |
| pytest | ≥ 7.0.0 | 测试框架 |

---

## 12. 关键注意事项

1. **风险评估**: 系统给出的风险分是启发式评估，不等同于真实后验概率，需结合元数据、视觉证据和来源信息综合判断。
2. **文档类图片**: 对于文档、流程图、截图等类型的图片，系统会自动降低视觉分数的权重。
3. **元数据优先级**: 如果图片包含明确的 AI 生成元数据，这是最强的证据，会直接判定为高风险。
4. **相机元数据**: 如果图片包含相机或拍摄链路元数据，系统会降低风险等级。

---

## 13. 扩展开发指南

### 添加新的分析模块
1. 在 `modules/` 目录下创建新模块文件
2. 在 `AnalysisService` 中集成新模块
3. 更新 API 路由（如需要）
4. 更新测试用例

### 添加新的导出格式
1. 在 `ExportService` 中添加新的生成方法
2. 更新 `export()` 方法的格式判断逻辑
3. 更新前端的格式选择选项

---

*本 Code Wiki 最后更新时间: 2026-05-20*
