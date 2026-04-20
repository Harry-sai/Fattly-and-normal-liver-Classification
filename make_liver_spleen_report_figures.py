from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FEATURES_CSV = Path("artifacts/liver_spleen_features.csv")
FIGURES_DIR = Path("artifacts/report_figures")
RUN_DIRS = [
    Path("results/liver_spleen_dual_both"),
    Path("results/liver_spleen_dual_liver_only"),
    Path("results/liver_spleen_dual_spleen_only"),
]


def plot_feature_figures(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for label, class_name in [(0, "normal"), (1, "fatty_liver")]:
        subset = df[df["label"] == label]
        plt.hist(subset["mean_ratio"], bins=25, alpha=0.55, label=class_name)
    plt.xlabel("Liver / Spleen Mean Intensity Ratio")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_ratio_histogram.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    colors = df["label"].map({0: "tab:blue", 1: "tab:orange"})
    plt.scatter(df["spleen_mean"], df["liver_mean"], c=colors, alpha=0.7)
    plt.xlabel("Spleen Mean Intensity")
    plt.ylabel("Liver Mean Intensity")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "liver_vs_spleen_scatter.png")
    plt.close()


def plot_ablation_summary():
    rows = []
    for run_dir in RUN_DIRS:
        oof_path = run_dir / "oof_predictions.csv"
        if not oof_path.exists():
            continue
        df = pd.read_csv(oof_path)
        auc = float("nan")
        if df["label"].nunique() > 1:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(df["label"], df["prob"])
        rows.append({"run": run_dir.name, "auc": auc})

    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(FIGURES_DIR / "ablation_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["run"], summary_df["auc"])
    plt.ylabel("OOF AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_auc_bar.png")
    plt.close()


def main():
    df = pd.read_csv(FEATURES_CSV)
    plot_feature_figures(df)
    plot_ablation_summary()
    print(f"Saved report figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
