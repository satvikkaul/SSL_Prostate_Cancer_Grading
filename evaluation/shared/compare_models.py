"""
Compare all available model evaluation outputs.

Reads metrics from:
  - ./output/baseline/evaluation_metrics.txt
  - ./output/cae/evaluation_metrics.txt
  - ./output/simclr/evaluation_metrics.txt
  - ./output/moco/evaluation_metrics.txt
"""

import os
import re

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ["NC", "G3", "G5", "G4"]
OUTPUT_DIR = "./output"
COMPARISON_FILE = os.path.join(OUTPUT_DIR, "model_comparison.csv")
COMPARISON_PLOT = os.path.join(OUTPUT_DIR, "model_comparison_plots.png")
COMPARISON_REPORT = os.path.join(OUTPUT_DIR, "comparison_report.txt")

MODEL_SPECS = [
    ("Baseline (No SSL)", "./output/baseline/evaluation_metrics.txt", "#ff9999"),
    ("CAE-SSL", "./output/cae/evaluation_metrics.txt", "#66b3ff"),
    ("SimCLR-SSL", "./output/simclr/evaluation_metrics.txt", "#99ff99"),
    ("MoCo-SSL", "./output/moco/evaluation_metrics.txt", "#ffcc80"),
]


def empty_metrics():
    return {
        "overall_accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "cohens_kappa": np.nan,
        "class_metrics": {
            cls: {"precision": np.nan, "recall": np.nan, "f1": np.nan}
            for cls in CLASS_NAMES
        },
    }


def parse_metrics_file(filepath):
    metrics = empty_metrics()
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read()

    acc_match = re.search(r"accuracy\s+(\d+\.\d+)\s+\d+", content)
    if acc_match:
        metrics["overall_accuracy"] = float(acc_match.group(1))

    macro_match = re.search(r"macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content)
    if macro_match:
        metrics["macro_f1"] = float(macro_match.group(3))

    weighted_match = re.search(r"weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content)
    if weighted_match:
        metrics["weighted_f1"] = float(weighted_match.group(3))

    kappa_match = re.search(r"Cohen's Kappa.*?(\d+\.\d+)", content)
    if kappa_match:
        metrics["cohens_kappa"] = float(kappa_match.group(1))

    for cls in CLASS_NAMES:
        cls_match = re.search(rf"^{cls}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content, re.MULTILINE)
        if cls_match:
            metrics["class_metrics"][cls] = {
                "precision": float(cls_match.group(1)),
                "recall": float(cls_match.group(2)),
                "f1": float(cls_match.group(3)),
            }

    return metrics


def load_all_metrics():
    loaded = []
    for model_name, metrics_path, color in MODEL_SPECS:
        metrics = parse_metrics_file(metrics_path)
        if metrics is None:
            print(f"Skipping {model_name}: missing {metrics_path}")
            continue
        loaded.append((model_name, metrics_path, color, metrics))
        print(f"Loaded {model_name}: {metrics_path}")
    return loaded


def build_tables(loaded_metrics):
    overall_rows = []
    class_rows = []

    for model_name, metrics_path, color, metrics in loaded_metrics:
        overall_rows.append(
            {
                "Model": model_name,
                "Metrics Path": metrics_path,
                "Overall Accuracy": metrics["overall_accuracy"],
                "Macro F1-Score": metrics["macro_f1"],
                "Weighted F1-Score": metrics["weighted_f1"],
                "Cohen's Kappa": metrics["cohens_kappa"],
                "Color": color,
            }
        )

        for cls in CLASS_NAMES:
            class_rows.append(
                {
                    "Model": model_name,
                    "Class": cls,
                    "Precision": metrics["class_metrics"][cls]["precision"],
                    "Recall": metrics["class_metrics"][cls]["recall"],
                    "F1-Score": metrics["class_metrics"][cls]["f1"],
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def plot_comparison(overall_df, class_df):
    fig = plt.figure(figsize=(16, 12))
    colors = overall_df["Color"].tolist()
    labels = overall_df["Model"].tolist()

    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(range(len(labels)), overall_df["Overall Accuracy"], color=colors, edgecolor="black")
    ax1.set_title("Overall Accuracy")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_ylim([0, 1])
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom")

    ax2 = plt.subplot(3, 3, 2)
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width / 2, overall_df["Macro F1-Score"], width, label="Macro F1", color="#ef9a9a", edgecolor="black")
    ax2.bar(x + width / 2, overall_df["Weighted F1-Score"], width, label="Weighted F1", color="#90caf9", edgecolor="black")
    ax2.set_title("F1 Scores")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=10, ha="right")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    bars = ax3.bar(range(len(labels)), overall_df["Cohen's Kappa"], color=colors, edgecolor="black")
    ax3.set_title("Cohen's Kappa")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=10, ha="right")
    ax3.set_ylim([0, 1])
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom")

    for idx, cls in enumerate(CLASS_NAMES):
        ax = plt.subplot(3, 3, 4 + idx)
        cls_data = class_df[class_df["Class"] == cls]
        bars = ax.bar(range(len(cls_data)), cls_data["Recall"], color=colors[: len(cls_data)], edgecolor="black")
        ax.set_title(f"{cls} Recall")
        ax.set_xticks(range(len(cls_data)))
        ax.set_xticklabels(cls_data["Model"].tolist(), rotation=10, ha="right")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=150, bbox_inches="tight")


def write_report(overall_df, class_df):
    best_idx = overall_df["Overall Accuracy"].idxmax()
    best_row = overall_df.loc[best_idx]

    baseline_row = overall_df[overall_df["Model"] == "Baseline (No SSL)"]
    baseline_acc = baseline_row["Overall Accuracy"].iloc[0] if not baseline_row.empty else np.nan

    with open(COMPARISON_REPORT, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("OVERALL METRICS COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(overall_df.drop(columns=["Color"]).to_string(index=False))
        f.write("\n\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. Best Performing Model: {best_row['Model']} ({best_row['Overall Accuracy']:.3f} accuracy)\n")
        if not np.isnan(baseline_acc):
            for _, row in overall_df.iterrows():
                if row["Model"] == "Baseline (No SSL)":
                    continue
                improvement = ((row["Overall Accuracy"] - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else np.nan
                f.write(f"2. {row['Model']} Improvement vs Baseline: {improvement:+.1f}%\n")

        f.write("\nPER-CLASS PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        for cls in CLASS_NAMES:
            f.write(f"\n{cls} Class:\n")
            cls_data = class_df[class_df["Class"] == cls]
            for _, row in cls_data.iterrows():
                f.write(
                    f"  {row['Model']:18s}: Recall={row['Recall']:.3f}, "
                    f"Precision={row['Precision']:.3f}, F1={row['F1-Score']:.3f}\n"
                )


def main():
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)

    loaded_metrics = load_all_metrics()
    if len(loaded_metrics) < 2:
        raise RuntimeError("Need at least two evaluation_metrics.txt files to compare models.")

    overall_df, class_df = build_tables(loaded_metrics)
    overall_df.to_csv(COMPARISON_FILE, index=False)
    class_df.to_csv(COMPARISON_FILE.replace(".csv", "_per_class.csv"), index=False)
    plot_comparison(overall_df, class_df)
    write_report(overall_df, class_df)

    print(f"Saved comparison table: {COMPARISON_FILE}")
    print(f"Saved comparison plot: {COMPARISON_PLOT}")
    print(f"Saved comparison report: {COMPARISON_REPORT}")


if __name__ == "__main__":
    main()
