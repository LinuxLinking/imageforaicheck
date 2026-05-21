# 小样本训练与特征提取完整指南

**当前情况**：约 550 个样本，其中仅 3-6 个是 AI 生成图片

---

## 📋 目录
1. [快速开始：用你的样本重新训练模型](#快速开始用你的样本重新训练模型)
2. [方案详解](#方案详解)
3. [YOLOv8 特征说明](#yolov8-特征说明)
4. [完整操作步骤](#完整操作步骤)

---

## 🚀 快速开始：用你的样本重新训练模型

### 准备工作

```bash
cd backend

# 创建样本文件夹结构
mkdir -p samples/ai       # 放入你的 3-6 张 AI 生成图
mkdir -p samples/real     # 放入你的真实照片
mkdir -p models           # 存放训练好的模型
```

### 一行命令训练（包含 YOLO 特征）

```bash
# 训练模型（自动扩增 AI 样本，包含 YOLO 特征）
python train_small_sample.py \
  --ai samples/ai \
  --real samples/real \
  --out models/my_model.json
```

### 训练完成后

训练好的模型会保存为 `models/my_model.json`，然后在应用中配置使用此模型即可。

---

## 🎯 方案详解

### 方案对比

| 方案 | 需要样本 | 难度 | 效果 | 推荐度 |
|------|---------|------|------|--------|
| **方案A：规则系统校准** | 3-6个AI + 50+真实 | ⭐ | 中等 | ⭐⭐⭐ |
| **方案B：重新训练模型（推荐）** | 3-6个AI + 100+真实 | ⭐⭐ | 好 | ⭐⭐⭐⭐⭐ |
| **方案C：预训练模型融合** | 无需 | ⭐⭐ | 好 | ⭐⭐⭐⭐ |

---

## 🤖 YOLOv8 特征说明

### 已集成的 YOLOv8 特征

我们已经把 YOLOv8 特征提取集成到项目中！以下是新增的 5 个特征：

| 特征名 | 说明 | AI图片特点 |
|--------|------|-----------|
| `yolo_num_detections_n` | 检测到的物体数量 | 可能偏少或偏多 |
| `yolo_avg_confidence_n` | 平均检测置信度 | 通常偏低 |
| `yolo_max_confidence_n` | 最高检测置信度 | 可能较低 |
| `yolo_std_confidence_n` | 置信度标准差 | 可能异常 |
| `yolo_anomaly_score` | 综合异常分数 | AI图片通常更高 |

### YOLOv8 特征提取原理

```python
# 已在 object_detector.py 中实现
extract_yolo_features(image_path) -> {
    num_detections: 检测到几个物体,
    avg_confidence: 平均置信度,
    max_confidence: 最高置信度,
    std_confidence: 置信度标准差,
    yolo_anomaly_score: 综合异常分,
}
```

---

## 📝 完整操作步骤

### 第 1 步：准备样本

```
backend/
├── samples/
│   ├── ai/
│   │   ├── ai_1.jpg
│   │   ├── ai_2.png
│   │   ├── ai_3.webp
│   │   └── (放入你的 AI 生成图，3-6张即可)
│   └── real/
│       ├── photo_1.jpg
│       ├── photo_2.jpg
│       └── (放入 100-500 张真实照片)
└── ...
```

### 第 2 步：训练模型（关键步骤）

#### 选项 1：使用 YOLO 特征（推荐）

```bash
cd backend

python train_small_sample.py \
  --ai samples/ai \
  --real samples/real \
  --out models/ai_detector_with_yolo.json
```

#### 选项 2：不使用 YOLO（如果速度太慢）

```bash
python train_small_sample.py \
  --ai samples/ai \
  --real samples/real \
  --out models/ai_detector_simple.json \
  --no-yolo
```

#### 选项 3：调整扩增倍数

```bash
# 每张 AI 样本扩增 20 倍（默认 15 倍）
python train_small_sample.py \
  --ai samples/ai \
  --real samples/real \
  --out models/ai_detector_augmented.json \
  --augment 20
```

### 第 3 步：查看训练结果

训练过程会输出：

```
=======================================================
AI 图像检测 - 小样本训练工具
=======================================================

【第1步】加载样本
从 samples/ai 加载了 5 张图片
从 samples/real 加载了 550 张图片

【第2步】数据增强 AI 样本
  每张 AI 样本扩增 15 倍
  扩增后 AI 样本: 75 张

【第3步】特征提取
  样本数: 625
  特征维度: 14 (包含 5 个 YOLO 特征)
  AI 样本数: 75
  真实样本数: 550

【第4步】训练模型...

【第5步】评估
  准确率: 92.5%
  精确率: 89.2%
  召回率: 85.7%

【第6步】保存模型
✓ 模型已保存到: models/ai_detector_with_yolo.json
```

### 第 4 步：在应用中使用模型

#### 修改代码，加载训练好的模型

在 `app.py` 或相关初始化代码中：

```python
# 方法 1：通过配置传入
service = AnalysisService(ml_model_path="models/ai_detector_with_yolo.json")

# 方法 2：或在 app.py 中修改
if __name__ == '__main__':
    # 创建服务时传入模型路径
    from imageforai.services.analysis_service import AnalysisService
    service = AnalysisService(ml_model_path="models/ai_detector_with_yolo.json")
    # ... 启动服务
```

---

## 🔧 工具文件说明

| 文件 | 用途 |
|------|------|
| `backend/train_small_sample.py` | **小样本训练工具（新）** |
| `backend/calibrate_model.py` | 规则系统校准工具 |
| `backend/augment_samples.py` | 数据增强工具 |
| `backend/src/imageforai/modules/object_detector.py` | YOLOv8 检测器（已更新） |
| `backend/src/imageforai/ml/feature_extractor.py` | 特征提取器（已更新） |

---

## 📊 如何选择方案？

### 只有 3-6 个 AI 样本 → **选择方案B（重新训练）**

**理由**：
1. 我们已经有训练框架
2. 可以使用数据扩增
3. 自动包含 YOLOv8 特征
4. 效果通常比纯规则好

### 样本极少（<3个）→ **先用方案A（校准），再逐步过渡**

### 想要最好效果 → **方案B + 方案C（预训练融合）**

---

## 💡 进阶技巧

### 技巧 1：特征分析

想知道哪些特征最重要？使用校准工具：

```bash
python calibrate_model.py \
  --ai samples/ai \
  --real samples/real
```

查看输出中的特征重要性分析！

### 技巧 2：比较有/无 YOLO 的效果

```bash
# 训练有 YOLO 的模型
python train_small_sample.py --ai samples/ai --real samples/real --out models/with_yolo.json

# 训练无 YOLO 的模型
python train_small_sample.py --ai samples/ai --real samples/real --out models/no_yolo.json --no-yolo

# 对比两个模型的准确率！
```

### 技巧 3：持续优化

上线后收集误判样本，定期重新训练：

```bash
# 新增样本到文件夹
mkdir -p samples/new_ai
mkdir -p samples/new_real

# 重新训练
python train_small_sample.py \
  --ai samples/ai \
  --real samples/real \
  --out models/updated_model.json
```

---

## ❓ 常见问题

### Q1: YOLOv8 是自动使用的吗？

**A**: 是的！
- 训练时，`train_small_sample.py` 默认包含 YOLO 特征
- 推理时，`MLClassifier` 也会自动使用相应的特征

### Q2: 我的样本太少，会过拟合吗？

**A**: 我们有防止过拟合的措施：
1. 使用 L2 正则化（`--l2 0.5`）
2. 数据扩增
3. 逻辑回归本身比较稳定

### Q3: 训练速度慢怎么办？

**A**: 可以：
1. 使用 `--no-yolo` 禁用 YOLO（更快，略降低准确率）
2. 减少真实样本的数量（比如只取 100-200 张）
3. 减少扩增倍数（`--augment 8`）

### Q4: 如何判断模型好坏？

**A**: 看训练输出：
- `准确率` 高（>85% 比较理想）
- `召回率` 高（AI 样本能被找到）
- `精确率` 高（误报少）

---

## 🎉 下一步

1. **立即开始**：把你的样本放入 `samples/ai` 和 `samples/real`
2. **运行训练**：执行 `python train_small_sample.py ...`
3. **集成模型**：在应用中配置使用训练好的模型
4. **持续优化**：收集反馈，定期更新模型

---

需要帮助？查看代码中的注释或运行：

```bash
python train_small_sample.py --help
```
