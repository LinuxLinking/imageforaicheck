import argparse
import csv
import os
import tempfile
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from .feature_extractor import DEFAULT_FEATURE_NAMES, extract_feature_vector


def iter_image_files(root_dir: str) -> List[str]:
    paths: List[str] = []
    for name in sorted(os.listdir(root_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            paths.append(os.path.join(root_dir, name))
    return paths


def downscale_if_needed(image_path: str, max_side: int, tmp_dir: str) -> Tuple[str, bool]:
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) <= max_side:
            return image_path, False

        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        out_path = os.path.join(tmp_dir, f"{os.path.basename(image_path)}.jpg")
        img.save(out_path, format="JPEG", quality=92)
        return out_path, True


def write_label_sheet(
    images_dir: str,
    out_csv: str,
    max_side: int = 1024,
    limit: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="imageforai_label_")
    files = iter_image_files(images_dir)
    if limit is not None:
        files = files[: int(limit)]

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "raw_ai_probability",
                "raw_confidence",
                "downscaled_for_scoring",
                "width",
                "height",
                "created_at",
                "notes",
            ],
        )
        writer.writeheader()

        for path in files:
            score_path, scaled = downscale_if_needed(path, max_side=max_side, tmp_dir=tmp_dir)
            vector, base = extract_feature_vector(score_path, feature_names=DEFAULT_FEATURE_NAMES)
            writer.writerow(
                {
                    "path": path,
                    "label": "",
                    "raw_ai_probability": base.get("ai_probability_raw", 0.0),
                    "raw_confidence": base.get("confidence_raw", 0.0),
                    "downscaled_for_scoring": "1" if scaled else "0",
                    "width": base.get("width", 0),
                    "height": base.get("height", 0),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    write_label_sheet(
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        max_side=args.max_side,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

