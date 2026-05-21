# 小样本校准指南

**当前情况：** 约 550 个样本，其中仅 3-6 个是 AI 生成图片

---

## 🚀 快速开始（推荐执行顺序）

### 第 1 步：校准现有规则系统（立即可用）

使用你现有的 3-6 个 AI 样本和 550 个真实样本来优化现有公式：

```bash
cd backend

# 准备样本目录
mkdir -p samples/ai_samples    # 放你的 3-6 张AI图
mkdir -p samples/real_samples  # 放你的真实照片

# 运行校准工具
python calibrate_model.py \
  --ai-dir samples/ai_samples \
  --real-dir samples/real_samples \
  --output calibration_report.json
```

**这会生成：**
- AI 样本和真实样本的得分分布
- 推荐的阈值
- 特征重要性分析
- 配置建议

---

### 第 2 步：数据增强 AI 样本（扩充训练数据）

用 3-6 个 AI 样本生成更多变体：

```bash
# 方式一：预设模式（推荐，更可控）
python augment_samples.py \
  --input-dir samples/ai_samples \
  --preset \
  --output-dir augmented_samples

# 方式二：随机增强模式
python augment_samples.py \
  --input-dir samples/ai_samples \
  --num-variations 20 \
  --output-dir augmented_samples
```

**效果：** 3 个 AI 样本 → **90-180 个变体**

---

### 第 3 步（可选）：集成预训练模型

如果需要更好的效果，可以集成 SOTA 预训练模型：

```bash
# 安装依赖
pip install transformers torch pillow
```

然后在代码中使用：

```python
from imageforai.modules.pretrained_detector import create_fusion_detector

detector = create_fusion_detector()
result = detector.detect("image.jpg")
```

---

## 📊 三个方案详解

### 方案一：校准现有规则系统（最推荐 ✅）

**优点：**
- ✅ 无需额外数据
- ✅ 立即可用
- ✅ 可解释性强
- ✅ 保持现有架构

**工作原理：**
1. 用现有样本分析特征分布
2. 找出区分度最高的特征
3. 自动优化权重和阈值

**使用示例：**
```python
# 查看校准报告
cat calibration_report.json

# 根据报告手动调整权重
# 编辑 file: backend/src/imageforai/modules/ai_detector.py
```

---

### 方案二：数据增强 AI 样本

**优点：**
- ✅ 快速扩充样本量
- ✅ 不改变模型结构
- ✅ 保留原图特征

**增强方式：**
- 亮度、对比度、色彩调整
- 随机裁剪、旋转
- 轻微模糊
- 缩放

**注意：** 增强后的样本仅用于分析特征，不建议直接训练模型（3个样本太少）

---

### 方案三：集成预训练模型

**优点：**
- ✅ 使用业界最佳模型
- ✅ 无需自己训练
- ✅ 效果通常更好

**推荐模型：**
- FalconAI/ImageGuardian（轻量，效果好）
- 其他 HuggingFace 上的 AI 检测模型

---

## 🎯 具体操作步骤

### 准备你的样本

```
your_project/
├── samples/
│   ├── ai_samples/      # 放 3-6 张 AI 生成图
│   │   ├── ai_1.jpg
│   │   ├── ai_2.png
│   │   └── ...
│   └── real_samples/    # 放 50-100 张真实照片
│       ├── real_1.jpg
│       └── ...
└── ...
```

---

### 运行完整流程

```bash
# 1. 分析现有样本
python calibrate_model.py \
  --ai-dir samples/ai_samples \
  --real-dir samples/real_samples

# 2. 查看报告，决定是否调整权重
cat calibration_report.json

# 3. 增强 AI 样本（可选）
python augment_samples.py \
  --input-dir samples/ai_samples \
  --preset

# 4. 用增强后的样本重新分析（可选）
python calibrate_model.py \
  --ai-dir augmented_samples \
  --real-dir samples/real_samples
```

---

## 🔧 如何应用校准结果

### 方法一：手动调整权重

根据校准报告编辑：
`backend/src/imageforai/modules/ai_detector.py`

```python
# 在 __init__ 中调整
self.features_weights = {
    # 根据报告调整这些值
    'color_entropy': 0.12,
    'noise_level': 0.12,
    'yolo_anomaly_score': 0.15,  # 提高这个
    'metadata_score': 0.12,      # 或这个
    # ...
}
```

---

### 方法二：调整阈值

根据报告调整风险阈值：
`backend/src/imageforai/services/analysis_service.py`

```python
# 找到这些值并调整
if ai_probability >= 0.75:  # 根据报告改为 0.6 或 0.8
    level = "high"
elif ai_probability >= 0.4:  # 根据报告调整
    level = "medium"
```

---

## 💡 额外建议

### 1. 收集更多 AI 样本（长期）

即使只有 10-20 个，效果也会明显提升。

**来源建议：**
- Midjourney/DALL-E/Stable Diffusion 生成的图
- 公开数据集（如 GenImage）
- 自己生成一些

---

### 2. 使用 YOLO 特征

我们已经把 YOLO 集成进去了，确保：
- `yolov8n.pt` 文件在项目根目录
- 查看校准报告中 YOLO 特征的重要性

---

### 3. 持续迭代

1. 先用当前方案上线
2. 收集用户反馈（判断错误的案例）
3. 把错误案例加入样本集
4. 定期重新校准

---

## 📁 已创建的工具文件

| 文件 | 用途 |
|------|------|
| `backend/calibrate_model.py` | 样本校准工具 |
| `backend/augment_samples.py` | 数据增强工具 |
| `backend/src/imageforai/modules/pretrained_detector.py` | 预训练模型集成 |

---

## ❓ 常见问题

**Q: 只有 3 个 AI 样本，能有效吗？**

A: 可以！因为我们不是重新训练模型，而是：
- 分析现有样本的特征
- 校准规则和阈值
- 结合预训练模型

---

**Q: 增强后的样本会影响检测吗？**

A: 增强样本只用于分析特征分布，不会直接用于检测。检测时仍使用原图。

---

**Q: 推荐先用哪个方案？**

A: 按顺序：
1. **先用方案一**（校准现有规则）
2. **再用方案二**（数据增强，辅助分析）
3. **最后方案三**（如果效果还不够）
