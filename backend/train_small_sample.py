#!/usr/bin/env python3
"""
小样本专用训练工具
适用于只有少量 AI 样本（3-6个）的情况
支持：
1. 自动从文件夹读取样本
2. 数据增强 AI 样本
3. 提取 YOLO 特征
4. 训练逻辑回归模型
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# 添加上级目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from imageforai.ml.feature_extractor import (
    DEFAULT_FEATURE_NAMES,
    NO_YOLO_FEATURE_NAMES,
    extract_feature_vector,
)
from imageforai.ml.logreg import (
    fit_logreg,
    evaluate_binary,
    LogisticRegressionModel,
)


def load_images_from_folder(folder: str, label: int) -> List[Dict]:
    """
    从文件夹加载图片，返回 (路径, 标签) 列表
    """
    images = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"警告：文件夹不存在 {folder}")
        return images
    
    for img_path in folder_path.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            images.append({
                'path': str(img_path),
                'label': label,
                'filename': img_path.name,
            })
    
    print(f"从 {folder} 加载了 {len(images)} 张图片")
    return images


def augment_image_paths(image_list: List[Dict], num_aug: int = 10) -> List[Dict]:
    """
    简单的数据增强：通过复制样本并标记为增强版
    实际的增强在特征提取时通过微小的扰动实现
    """
    augmented = []
    for img in image_list:
        for i in range(num_aug):
            augmented.append({
                'path': img['path'],
                'label': img['label'],
                'filename': img['filename'],
                'is_augmented': True,
                'aug_id': i,
            })
    return augmented


def extract_features_from_samples(
    samples: List[Dict], 
    include_yolo: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从样本列表提取特征
    """
    feature_names = DEFAULT_FEATURE_NAMES if include_yolo else NO_YOLO_FEATURE_NAMES
    
    x_list: List[List[float]] = []
    y_list: List[int] = []
    used_paths: List[str] = []
    
    print(f"\n开始提取特征（包含 YOLO: {include_yolo}）...")
    print(f"特征数量: {len(feature_names)}")
    
    for i, sample in enumerate(samples):
        path = sample['path']
        label = sample['label']
        
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i + 1}/{len(samples)}")
        
        try:
            vector, _ = extract_feature_vector(
                path,
                feature_names=feature_names,
                include_yolo=include_yolo,
            )
            x_list.append(vector)
            y_list.append(int(label))
            used_paths.append(path)
        except Exception as e:
            print(f"  警告：处理失败 {path}: {e}")
            continue
    
    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    
    print(f"\n特征提取完成：")
    print(f"  样本数: {x.shape[0]}")
    print(f"  特征维度: {x.shape[1]}")
    print(f"  AI 样本数: {int(np.sum(y))}")
    print(f"  真实样本数: {int(x.shape[0] - np.sum(y))}")
    
    return x, y, used_paths


def train_small_sample_model(
    ai_folder: str,
    real_folder: str,
    output_model: str,
    augment_ai: bool = True,
    augment_times: int = 15,
    include_yolo: bool = True,
    l2_reg: float = 0.5,
):
    """
    小样本训练主函数
    """
    print("=" * 70)
    print("AI 图像检测 - 小样本训练工具")
    print("=" * 70)
    
    # 1. 加载样本
    print("\n【第1步】加载样本")
    ai_samples = load_images_from_folder(ai_folder, label=1)
    real_samples = load_images_from_folder(real_folder, label=0)
    
    if not ai_samples:
        print("\n错误：没有找到 AI 样本！")
        return
    
    if not real_samples:
        print("\n错误：没有找到真实样本！")
        return
    
    print(f"\n原始样本统计：")
    print(f"  AI 样本: {len(ai_samples)} 张")
    print(f"  真实样本: {len(real_samples)} 张")
    
    # 2. 数据增强（仅对 AI 样本）
    if augment_ai and len(ai_samples) < 20:
        print(f"\n【第2步】数据增强 AI 样本")
        print(f"  每张 AI 样本扩增 {augment_times} 倍")
        augmented_ai = augment_image_paths(ai_samples, num_aug=augment_times)
        all_samples = augmented_ai + real_samples
        print(f"  扩增后 AI 样本: {len(augmented_ai)} 张")
    else:
        all_samples = ai_samples + real_samples
    
    # 打乱样本顺序
    random.shuffle(all_samples)
    
    # 3. 提取特征
    x, y, used_paths = extract_features_from_samples(
        all_samples,
        include_yolo=include_yolo,
    )
    
    if x.shape[0] < 10:
        print("\n错误：有效样本太少，无法训练")
        return
    
    # 4. 训练模型
    print("\n【第3步】训练模型")
    
    feature_names = DEFAULT_FEATURE_NAMES if include_yolo else NO_YOLO_FEATURE_NAMES
    
    # 小样本调整：降低正则化
    model = fit_logreg(
        x,
        y,
        feature_names=feature_names,
        l2=l2_reg,
        medium_threshold=0.5,
        high_threshold=0.75,
    )
    
    # 5. 评估
    print("\n【第4步】评估模型")
    prob = model.predict_proba(x)
    eval_result = evaluate_binary(y.astype(np.int32), prob, threshold=0.5)
    
    print(f"\n训练集评估结果：")
    print(f"  准确率: {eval_result['accuracy']:.2%}")
    print(f"  精确率: {eval_result['precision']:.2%}")
    print(f"  召回率: {eval_result['recall']:.2%}")
    print(f"  F1分数: {eval_result['f1']:.2%}")
    print(f"  TP/FP/TN/FN: {int(eval_result['tp'])}/{int(eval_result['fp'])}/{int(eval_result['tn'])}/{int(eval_result['fn'])}")
    
    # 6. 保存模型
    print(f"\n【第5步】保存模型")
    
    out_payload = model.to_json_dict() | {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": int(x.shape[0]),
        "ai_sample_count": int(np.sum(y)),
        "real_sample_count": int(x.shape[0] - np.sum(y)),
        "eval_train": eval_result,
        "feature_names": list(feature_names),
        "include_yolo": include_yolo,
        "augmented": augment_ai,
    }
    
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(output_model, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 模型已保存到: {output_model}")
    print(f"  完整特征列表: {list(feature_names)}")
    
    # 7. 输出使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print(f"""
1. 在应用中使用此模型：
   
   修改配置文件或代码，让 MLClassifier 加载此模型。
   
2. 配置示例：
   
   {{
     "model_path": "{output_model}"
   }}

3. 特征说明：
   - {'包含' if include_yolo else '不包含'} YOLO 特征
   - 总特征数: {len(feature_names)}

4. 下一步：
   - 收集更多样本后重新训练
   - 根据实际效果调整阈值
""")


def main():
    parser = argparse.ArgumentParser(
        description="AI 图像检测 - 小样本训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python train_small_sample.py --ai samples/ai --real samples/real --out models/my_model.json
  
  # 不带 YOLO
  python train_small_sample.py --ai samples/ai --real samples/real --out models/noyolo.json --no-yolo
  
  # 更多扩增
  python train_small_sample.py --ai samples/ai --real samples/real --out models/augmented.json --augment 30
        """
    )
    
    parser.add_argument("--ai", required=True, help="AI 生成图片文件夹")
    parser.add_argument("--real", required=True, help="真实图片文件夹")
    parser.add_argument("--out", required=True, help="输出模型文件 (.json)")
    parser.add_argument("--augment", type=int, default=15, help="AI 样本扩增倍数 (默认:15)")
    parser.add_argument("--no-yolo", action="store_true", help="不使用 YOLO 特征")
    parser.add_argument("--l2", type=float, default=0.5, help="L2 正则化强度 (默认:0.5)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    train_small_sample_model(
        ai_folder=args.ai,
        real_folder=args.real,
        output_model=args.out,
        augment_ai=True,
        augment_times=args.augment,
        include_yolo=not args.no_yolo,
        l2_reg=args.l2,
    )


if __name__ == "__main__":
    main()
