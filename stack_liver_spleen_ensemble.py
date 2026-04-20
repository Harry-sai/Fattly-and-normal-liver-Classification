import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Fold-aware stacking and calibration for liver+spleen classifiers.")
    parser.add_argument(
        "--masked-oof",
        default="classification/masked_efficientnetb0/v4_stabilized/oof_predictions.csv",
    )
    parser.add_argument(
        "--hybrid-oof",
        default="classification/hybrid_densenet_late_fusion/v2_class/oof_predictions.csv",
    )
    parser.add_argument(
        "--results-dir",
        default="classification/stacked_ensemble/v1",
    )
    return parser.parse_args()


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.uint8)
    probs = np.asarray(probs, dtype=np.float64)
    auc = float(roc_auc_score(labels, probs))
    brier = float(brier_score_loss(labels, probs))
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    ll = float(log_loss(labels, clipped))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[best_idx])
    preds = (probs >= best_thr).astype(np.uint8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "auc": auc,
        "brier": brier,
        "logloss": ll,
        "acc": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_thr": best_thr,
    }


def load_merged(masked_path, hybrid_path):
    masked = pd.read_csv(masked_path)
    hybrid = pd.read_csv(hybrid_path)
    merged = masked.merge(
        hybrid[
            [
                "image_id",
                "class_name",
                "label",
                "fold",
                "hybrid_prob",
                "image_branch_prob",
                "tabular_branch_prob",
                "weighted_blend_prob",
                "average_blend_prob",
            ]
        ],
        on=["image_id", "class_name", "label", "fold"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def fold_logreg_stack(df, features):
    y = df["label"].to_numpy(dtype=np.uint8)
    folds = sorted(df["fold"].unique())
    probs = np.zeros(len(df), dtype=np.float64)
    coef_rows = []

    for fold in folds:
        tr = df["fold"] != fold
        va = df["fold"] == fold
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        )
        pipe.fit(df.loc[tr, features], y[tr])
        probs[va] = pipe.predict_proba(df.loc[va, features])[:, 1]

        clf = pipe.named_steps["clf"]
        coef_rows.append(
            {
                "fold": int(fold),
                "intercept": float(clf.intercept_[0]),
                **{feature: float(weight) for feature, weight in zip(features, clf.coef_[0])},
            }
        )

    return probs, pd.DataFrame(coef_rows)


def fold_weight_search(df, col_a, col_b):
    y = df["label"].to_numpy(dtype=np.uint8)
    folds = sorted(df["fold"].unique())
    probs = np.zeros(len(df), dtype=np.float64)
    weight_rows = []

    for fold in folds:
        tr = df["fold"] != fold
        va = df["fold"] == fold
        best = None
        tr_a = df.loc[tr, col_a].to_numpy(dtype=np.float64)
        tr_b = df.loc[tr, col_b].to_numpy(dtype=np.float64)

        for weight_a in np.linspace(0.0, 1.0, 101):
            blend = weight_a * tr_a + (1.0 - weight_a) * tr_b
            auc = roc_auc_score(y[tr], blend)
            if best is None or auc > best[1]:
                best = (float(weight_a), float(auc))

        weight_a = best[0]
        probs[va] = (
            weight_a * df.loc[va, col_a].to_numpy(dtype=np.float64)
            + (1.0 - weight_a) * df.loc[va, col_b].to_numpy(dtype=np.float64)
        )
        weight_rows.append(
            {
                "fold": int(fold),
                f"{col_a}_weight": weight_a,
                f"{col_b}_weight": float(1.0 - weight_a),
                "train_auc": best[1],
            }
        )

    return probs, pd.DataFrame(weight_rows)


def fold_isotonic_calibration(df, raw_probs, reference_col_name):
    y = df["label"].to_numpy(dtype=np.uint8)
    folds = sorted(df["fold"].unique())
    probs = np.zeros(len(df), dtype=np.float64)

    for fold in folds:
        tr = df["fold"] != fold
        va = df["fold"] == fold
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs[tr], y[tr])
        probs[va] = iso.predict(raw_probs[va])

    return probs


def summarize_candidates(df, candidate_map):
    labels = df["label"].to_numpy(dtype=np.uint8)
    rows = []
    for name, probs in candidate_map.items():
        metrics = compute_metrics(labels, probs)
        fold_aucs = []
        for _, fold_df in df.assign(candidate_prob=probs).groupby("fold"):
            fold_aucs.append(roc_auc_score(fold_df["label"], fold_df["candidate_prob"]))
        rows.append(
            {
                "candidate": name,
                **metrics,
                "fold_mean_auc": float(np.mean(fold_aucs)),
                "fold_std_auc": float(np.std(fold_aucs)),
            }
        )
    return pd.DataFrame(rows).sort_values(["auc", "logloss"], ascending=[False, True])


def main():
    args = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged(args.masked_oof, args.hybrid_oof)
    df.to_csv(out_dir / "merged_base_oof_predictions.csv", index=False)

    simple_features = ["model_prob", "weighted_blend_prob"]
    full_features = [
        "model_prob",
        "blend_prob",
        "logreg_prob",
        "hybrid_prob",
        "weighted_blend_prob",
        "average_blend_prob",
        "image_branch_prob",
        "tabular_branch_prob",
    ]

    simple_stack, simple_coef = fold_logreg_stack(df, simple_features)
    full_stack, full_coef = fold_logreg_stack(df, full_features)
    weight_search, weight_df = fold_weight_search(df, "blend_prob", "weighted_blend_prob")
    iso_weighted_blend = fold_isotonic_calibration(df, df["weighted_blend_prob"].to_numpy(dtype=np.float64), "weighted_blend_prob")
    iso_simple_stack = fold_isotonic_calibration(df, simple_stack, "simple_stack_prob")
    iso_full_stack = fold_isotonic_calibration(df, full_stack, "full_stack_prob")

    candidate_map = {
        "masked_model_prob": df["model_prob"].to_numpy(dtype=np.float64),
        "masked_blend_prob": df["blend_prob"].to_numpy(dtype=np.float64),
        "hybrid_weighted_blend_prob": df["weighted_blend_prob"].to_numpy(dtype=np.float64),
        "simple_logreg_stack_prob": simple_stack,
        "full_logreg_stack_prob": full_stack,
        "fold_weight_search_prob": weight_search,
        "iso_weighted_blend_prob": iso_weighted_blend,
        "iso_simple_logreg_stack_prob": iso_simple_stack,
        "iso_full_logreg_stack_prob": iso_full_stack,
    }

    pred_df = df[["image_id", "class_name", "label", "fold"]].copy()
    for name, probs in candidate_map.items():
        pred_df[name] = probs
    pred_df.to_csv(out_dir / "stacked_oof_predictions.csv", index=False)

    simple_coef.to_csv(out_dir / "simple_stack_coefficients.csv", index=False)
    full_coef.to_csv(out_dir / "full_stack_coefficients.csv", index=False)
    weight_df.to_csv(out_dir / "fold_weight_search_weights.csv", index=False)

    summary_df = summarize_candidates(df, candidate_map)
    summary_df.to_csv(out_dir / "stacked_summary.csv", index=False)

    best_auc_row = summary_df.sort_values("auc", ascending=False).iloc[0]
    best_cal_row = summary_df.sort_values(["logloss", "brier"], ascending=[True, True]).iloc[0]

    report_lines = []
    report_lines.append("# Stacked Ensemble Summary")
    report_lines.append("")
    report_lines.append(f"- Masked OOF source: `{args.masked_oof}`")
    report_lines.append(f"- Hybrid OOF source: `{args.hybrid_oof}`")
    report_lines.append(f"- Samples merged: `{len(df)}`")
    report_lines.append("")
    report_lines.append("## Best By AUC")
    report_lines.append("")
    report_lines.append(
        f"- `{best_auc_row['candidate']}`: "
        f"AUC `{best_auc_row['auc']:.4f}`, "
        f"Accuracy `{best_auc_row['acc']:.4f}`, "
        f"F1 `{best_auc_row['f1']:.4f}`, "
        f"fold mean AUC `{best_auc_row['fold_mean_auc']:.4f} ± {best_auc_row['fold_std_auc']:.4f}`"
    )
    report_lines.append("")
    report_lines.append("## Best By Calibration")
    report_lines.append("")
    report_lines.append(
        f"- `{best_cal_row['candidate']}`: "
        f"log loss `{best_cal_row['logloss']:.4f}`, "
        f"Brier `{best_cal_row['brier']:.4f}`, "
        f"AUC `{best_cal_row['auc']:.4f}`"
    )
    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append("- Stacking improves performance only if it preserves ranking between strong base predictors.")
    report_lines.append("- Isotonic calibration improves probability quality but may slightly reduce AUC because calibration is not designed to improve ranking.")
    report_lines.append("- For thesis reporting, use the highest-AUC stacked model if discrimination is the priority, and report the best-calibrated model separately if probability quality matters.")
    (out_dir / "stacked_summary.md").write_text("\n".join(report_lines))

    print(summary_df.to_string(index=False))
    print("")
    print(f"Best by AUC: {best_auc_row['candidate']} | AUC={best_auc_row['auc']:.4f}")
    print(f"Best by calibration: {best_cal_row['candidate']} | logloss={best_cal_row['logloss']:.4f} brier={best_cal_row['brier']:.4f}")


if __name__ == "__main__":
    main()
