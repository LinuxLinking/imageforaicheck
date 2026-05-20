import argparse
import csv
import os
import random
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..modules.ai_detector import AIDetector


def iter_image_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(root_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            files.append(os.path.join(root_dir, name))
    files.sort()
    return files


def downscale_if_needed(image_path: str, max_side: int, tmp_dir: str) -> Tuple[str, str, str, bool]:
    with Image.open(image_path) as img:
        fmt = img.format or ""
        w0, h0 = img.size
        if max(w0, h0) <= max_side:
            return image_path, fmt, f"{w0}x{h0}", False

        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        w1, h1 = img.size
        out_path = os.path.join(tmp_dir, f"{os.path.basename(image_path)}.jpg")
        img.save(out_path, format="JPEG", quality=92)
        return out_path, fmt, f"{w1}x{h1}", True


def bucket(prob: float) -> str:
    if prob >= 0.75:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def build_seed_label_map() -> Dict[str, str]:
    return {
        r"d:\ProgramData\imageforAI\images\imagesFordatas\A casual iPhone ....png": "1",
        r"d:\ProgramData\imageforAI\images\imagesFordatas\屏幕截图 2026-05-20 102026.png": "1",
        r"d:\ProgramData\imageforAI\images\imagesFordatas\屏幕截图 2026-05-20 102001.png": "1",
    }


def write_active_label_sheet(
    images_dir: str,
    out_csv: str,
    max_side: int = 1024,
    limit: Optional[int] = 2000,
    seed: int = 7,
    top_k: int = 300,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    all_files = iter_image_files(images_dir)

    seed_labels = build_seed_label_map()
    seed_files = [path for path in seed_labels.keys() if os.path.exists(path)]
    rnd = random.Random(seed)
    if limit is not None and len(all_files) > int(limit):
        files = rnd.sample(all_files, int(limit))
        files.extend(seed_files)
        files = sorted(set(files))
    else:
        files = sorted(set(all_files + seed_files))

    tmp_dir = tempfile.mkdtemp(prefix="imageforai_active_label_")
    detector = AIDetector()

    scored: List[Dict] = []
    for path in files:
        score_path, fmt, analysis_res, scaled = downscale_if_needed(path, max_side=max_side, tmp_dir=tmp_dir)
        ai = detector.detect(score_path)
        prob = float(ai.get("ai_probability", 0.0) or 0.0)
        conf = float(ai.get("confidence", 0.0) or 0.0)
        scored.append(
            {
                "path": path,
                "label": seed_labels.get(path, ""),
                "raw_ai_probability": prob,
                "raw_confidence": conf,
                "risk_bucket": bucket(prob),
                "downscaled_for_scoring": "1" if scaled else "0",
                "format": fmt,
                "analysis_resolution": analysis_res,
            }
        )

    scored_sorted = sorted(scored, key=lambda x: float(x["raw_ai_probability"]), reverse=True)
    review_set = set(item["path"] for item in scored_sorted[: max(int(top_k), 1)])

    now = datetime.now().isoformat(timespec="seconds")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "raw_ai_probability",
                "raw_confidence",
                "risk_bucket",
                "recommend_review",
                "downscaled_for_scoring",
                "format",
                "analysis_resolution",
                "created_at",
                "notes",
            ],
        )
        writer.writeheader()
        for item in scored_sorted:
            writer.writerow(
                {
                    **item,
                    "recommend_review": "1" if item["path"] in review_set else "0",
                    "created_at": now,
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=300)
    args = parser.parse_args()

    write_active_label_sheet(
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        max_side=args.max_side,
        limit=args.limit,
        seed=args.seed,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
