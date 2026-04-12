import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


DATASET_CSV = Path("artifacts/liver_spleen_dataset.csv")
FEATURES_CSV = Path("artifacts/liver_spleen_features.csv")
RANGE_JSON = Path("artifacts/liver_spleen_reference_ranges.json")
EPS = 1e-6


def load_grayscale(path):
    return np.array(Image.open(path).convert("L")).astype(np.float32)


def load_mask(path, target_shape):
    mask = np.array(Image.open(path).convert("L"))
    if mask.shape != target_shape:
        mask = np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))
    return (mask > 0).astype(np.uint8)


def erode_mask(mask, iterations=1):
    if ndimage is None or mask.sum() == 0:
        return mask
    eroded = ndimage.binary_erosion(mask.astype(bool), iterations=iterations)
    eroded = eroded.astype(np.uint8)
    return eroded if eroded.sum() > 32 else mask


def organ_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
        }

    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def make_reference_ranges(df):
    normal_df = df[df["label"] == 0].copy()
    if normal_df.empty:
        return {}

    cols = [
        "mean_ratio",
        "median_ratio",
        "mean_diff",
        "median_diff",
        "p25_diff",
        "p75_diff",
    ]

    summary = {}
    for col in cols:
        summary[col] = {
            "normal_p05": float(normal_df[col].quantile(0.05)),
            "normal_p25": float(normal_df[col].quantile(0.25)),
            "normal_p50": float(normal_df[col].quantile(0.50)),
            "normal_p75": float(normal_df[col].quantile(0.75)),
            "normal_p95": float(normal_df[col].quantile(0.95)),
        }

    ratio_lo = summary["mean_ratio"]["normal_p05"]
    ratio_hi = summary["mean_ratio"]["normal_p95"]
    diff_lo = summary["mean_diff"]["normal_p05"]
    diff_hi = summary["mean_diff"]["normal_p95"]
    heuristic_pred = ((df["mean_ratio"].between(ratio_lo, ratio_hi)) & (df["mean_diff"].between(diff_lo, diff_hi))).astype(int)
    heuristic_pred = 1 - heuristic_pred
    heuristic_acc = float((heuristic_pred == df["label"]).mean())

    summary["normal_range_rule"] = {
        "description": "Predict normal if liver-to-spleen mean_ratio and mean_diff both fall inside normal 5th-95th percentile ranges; otherwise fatty_liver.",
        "heuristic_accuracy_on_full_dataset": heuristic_acc,
        "ratio_range": [ratio_lo, ratio_hi],
        "diff_range": [diff_lo, diff_hi],
    }
    return summary


def main():
    df = pd.read_csv(DATASET_CSV)
    rows = []

    for row in df.itertuples(index=False):
        image = load_grayscale(row.image_path)
        liver_mask = erode_mask(load_mask(row.liver_mask_path, image.shape))
        spleen_mask = erode_mask(load_mask(row.spleen_mask_path, image.shape))

        liver_values = image[liver_mask > 0]
        spleen_values = image[spleen_mask > 0]

        features = {
            "image_id": row.image_id,
            "class_name": row.class_name,
            "label": int(row.label),
        }
        features.update(organ_stats(liver_values, "liver"))
        features.update(organ_stats(spleen_values, "spleen"))

        liver_mean = features["liver_mean"]
        spleen_mean = features["spleen_mean"]
        liver_median = features["liver_median"]
        spleen_median = features["spleen_median"]

        features.update(
            {
                "mean_ratio": float(liver_mean / (spleen_mean + EPS)),
                "median_ratio": float(liver_median / (spleen_median + EPS)),
                "mean_diff": float(liver_mean - spleen_mean),
                "median_diff": float(liver_median - spleen_median),
                "std_ratio": float(features["liver_std"] / (features["spleen_std"] + EPS)),
                "count_ratio": float(features["liver_count"] / (features["spleen_count"] + EPS)),
                "p25_diff": float(features["liver_p25"] - features["spleen_p25"]),
                "p50_diff": float(features["liver_p50"] - features["spleen_p50"]),
                "p75_diff": float(features["liver_p75"] - features["spleen_p75"]),
            }
        )
        rows.append(features)

    features_df = pd.DataFrame(rows)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_CSV, index=False)

    range_summary = make_reference_ranges(features_df)
    RANGE_JSON.write_text(json.dumps(range_summary, indent=2))

    print(f"Saved features to {FEATURES_CSV}")
    print(f"Saved attenuation reference summary to {RANGE_JSON}")


if __name__ == "__main__":
    main()
