import argparse
import csv
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from .feature_extractor import DEFAULT_FEATURE_NAMES, extract_feature_vector
from .logreg import evaluate_binary, fit_logreg, suggest_threshold_for_high_ratio


def load_labeled_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def parse_label(value: str) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    if value in {"0", "1"}:
        return int(value)
    lowered = value.lower()
    if lowered in {"ai", "true", "yes", "y"}:
        return 1
    if lowered in {"real", "false", "no", "n"}:
        return 0
    return None


def build_dataset(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_names = DEFAULT_FEATURE_NAMES
    x_list: List[List[float]] = []
    y_list: List[int] = []
    used_paths: List[str] = []

    for row in rows:
        label = parse_label(row.get("label", ""))
        if label is None:
            continue
        path = row.get("path", "")
        if not path or not os.path.exists(path):
            continue
        vector, _base = extract_feature_vector(path, feature_names=feature_names)
        x_list.append(vector)
        y_list.append(int(label))
        used_paths.append(path)

    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return x, y, used_paths


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = list(range(x.shape[0]))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    split = int(round(x.shape[0] * (1.0 - val_ratio)))
    train_idx = idx[:split]
    val_idx = idx[split:]
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    return x_train, y_train, x_val, y_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--l2", type=float, default=0.8)
    parser.add_argument("--target-high-ratio", type=float, default=0.2)
    parser.add_argument("--medium-threshold", type=float, default=0.6)
    parser.add_argument("--high-threshold", type=float, default=0.85)
    args = parser.parse_args()

    rows = load_labeled_rows(args.labels_csv)
    x, y, used_paths = build_dataset(rows)
    if x.shape[0] < 50:
        raise SystemExit(f"标注样本太少：{x.shape[0]}，建议至少 50 张起步")

    x_train, y_train, x_val, y_val = split_train_val(x, y, val_ratio=args.val_ratio, seed=args.seed)
    model = fit_logreg(
        x_train,
        y_train,
        feature_names=DEFAULT_FEATURE_NAMES,
        l2=args.l2,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
    )

    prob_train = model.predict_proba(x_train)
    prob_val = model.predict_proba(x_val)

    eval_train = evaluate_binary(y_train.astype(np.int32), prob_train, threshold=0.5)
    eval_val = evaluate_binary(y_val.astype(np.int32), prob_val, threshold=0.5)

    suggested_high = suggest_threshold_for_high_ratio(prob_val, target_ratio=args.target_high_ratio)
    suggested_medium = min(max(suggested_high - 0.15, 0.05), 0.95)

    out_payload = model.to_json_dict() | {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "labels_csv": os.path.abspath(args.labels_csv),
        "sample_count": int(x.shape[0]),
        "train_count": int(x_train.shape[0]),
        "val_count": int(x_val.shape[0]),
        "eval_train@0.5": eval_train,
        "eval_val@0.5": eval_val,
        "suggested_thresholds": {
            "target_high_ratio": float(args.target_high_ratio),
            "high_threshold": float(suggested_high),
            "medium_threshold": float(suggested_medium),
        },
    }

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    with open(args.out_model, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(args.out_model)
    print(json.dumps(out_payload["eval_val@0.5"], ensure_ascii=False, indent=2))
    print(json.dumps(out_payload["suggested_thresholds"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

