#!/usr/bin/env python3
"""
AI样本数据增强工具
利用有限的AI样本生成更多训练数据
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
from typing import List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class SampleAugmentor:
    """样本增强器"""
    
    def __init__(self):
        self.output_dir = "augmented_samples"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def augment_image(self, image_path: str, num_variations: int = 20) -> List[str]:
        """
        对单张图片进行数据增强
        
        Args:
            image_path: 原图路径
            num_variations: 生成多少张变体
            
        Returns:
            生成的图片路径列表
        """
        img = Image.open(image_path).convert('RGB')
        base_name = Path(image_path).stem
        generated_paths = []
        
        print(f"增强: {image_path} -> 生成 {num_variations} 张变体")
        
        for i in range(num_variations):
            # 随机组合多种增强方式
            aug_img = img.copy()
            
            # 1. 随机裁剪
            if random.random() > 0.3:
                aug_img = self._random_crop(aug_img)
            
            # 2. 亮度调整
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                aug_img = ImageEnhance.Brightness(aug_img).enhance(factor)
            
            # 3. 对比度调整
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)
            
            # 4. 色彩调整
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                aug_img = ImageEnhance.Color(aug_img).enhance(factor)
            
            # 5. 轻微模糊
            if random.random() > 0.7:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # 6. 轻微旋转
            if random.random() > 0.6:
                angle = random.uniform(-3, 3)
                aug_img = aug_img.rotate(angle, expand=True)
            
            # 7. 轻微缩放
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                w, h = aug_img.size
                new_w = int(w * scale)
                new_h = int(h * scale)
                aug_img = aug_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 保存
            output_path = os.path.join(self.output_dir, f"{base_name}_aug_{i+1}.jpg")
            aug_img.save(output_path, quality=90)
            generated_paths.append(output_path)
        
        return generated_paths
    
    def _random_crop(self, img: Image.Image) -> Image.Image:
        """随机裁剪"""
        w, h = img.size
        crop_ratio = random.uniform(0.85, 0.98)
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)
        
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        return img.crop((left, top, left + new_w, top + new_h))
    
    def augment_directory(self, input_dir: str, num_variations: int = 20) -> List[str]:
        """
        增强整个目录
        
        Args:
            input_dir: 输入目录
            num_variations: 每张图生成多少变体
            
        Returns:
            所有生成的图片路径
        """
        all_generated = []
        
        if not os.path.exists(input_dir):
            print(f"错误: 目录不存在 {input_dir}")
            return all_generated
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        for img_path in Path(input_dir).glob("*"):
            if img_path.suffix.lower() in image_extensions:
                generated = self.augment_image(str(img_path), num_variations)
                all_generated.extend(generated)
        
        print(f"\n完成! 共生成 {len(all_generated)} 张增强图片")
        print(f"保存位置: {os.path.abspath(self.output_dir)}")
        
        return all_generated
    
    def create_presets(self, image_path: str) -> List[str]:
        """
        创建预设的增强变体（非随机，更可控）
        适合小样本场景
        """
        img = Image.open(image_path).convert('RGB')
        base_name = Path(image_path).stem
        generated = []
        
        print(f"创建预设变体: {image_path}")
        
        presets = [
            ("bright", 0.85, 1.0, 1.0),   # 稍暗
            ("bright", 1.15, 1.0, 1.0),   # 稍亮
            ("contrast", 1.0, 0.85, 1.0), # 低对比度
            ("contrast", 1.0, 1.15, 1.0), # 高对比度
            ("color", 1.0, 1.0, 0.85),    # 低饱和
            ("color", 1.0, 1.0, 1.15),    # 高饱和
        ]
        
        for name, b, c, co in presets:
            aug_img = img.copy()
            aug_img = ImageEnhance.Brightness(aug_img).enhance(b)
            aug_img = ImageEnhance.Contrast(aug_img).enhance(c)
            aug_img = ImageEnhance.Color(aug_img).enhance(co)
            
            output_path = os.path.join(self.output_dir, f"{base_name}_preset_{name}.jpg")
            aug_img.save(output_path, quality=95)
            generated.append(output_path)
        
        return generated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI样本数据增强工具")
    parser.add_argument("--input-dir", required=True, help="AI样本目录")
    parser.add_argument("--num-variations", type=int, default=15, 
                       help="每张图生成多少变体")
    parser.add_argument("--output-dir", default="augmented_samples",
                       help="输出目录")
    parser.add_argument("--preset", action="store_true",
                       help="使用预设模式（非随机）")
    
    args = parser.parse_args()
    
    augmentor = SampleAugmentor()
    augmentor.output_dir = args.output_dir
    
    if args.preset:
        # 预设模式
        all_generated = []
        for img_path in Path(args.input_dir).glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                generated = augmentor.create_presets(str(img_path))
                all_generated.extend(generated)
        print(f"预设模式完成，生成 {len(all_generated)} 张")
    else:
        # 随机增强模式
        augmentor.augment_directory(args.input_dir, args.num_variations)


if __name__ == '__main__':
    main()
