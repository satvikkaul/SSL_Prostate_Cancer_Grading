"""
Compare all available model evaluation outputs.

By default, reads metrics from:
    - ./output/baseline/evaluation_metrics.txt
    - ./output/cae/evaluation_metrics.txt
    - ./output/simclr/evaluation_metrics.txt
    - ./output/moco/evaluation_metrics.txt

Use --base-dir and --output-dir to compare archived outputs stored elsewhere.
"""

import argparse
import os
import re

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ["NC", "G3", "G5", "G4"]
MODEL_COLORS = {
    "Baseline (No SSL)": "#ff9999",
    "CAE-SSL": "#66b3ff",
    "SimCLR-SSL": "#99ff99",
    "MoCo-SSL": "#ffcc80",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model evaluation outputs.")
    parser.add_argument(
        "--base-dir",
        default="./output",
        help="Directory containing baseline/cae/simclr/moco evaluation subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where comparison CSV/report/plot files will be written. Defaults to --base-dir.",
    )
    return parser.parse_args()


def build_model_specs(base_dir):
    return [
        ("Baseline (No SSL)", os.path.join(base_dir, "baseline", "evaluation_metrics.txt"), MODEL_COLORS["Baseline (No SSL)"]),
        ("CAE-SSL", os.path.join(base_dir, "cae", "evaluation_metrics.txt"), MODEL_COLORS["CAE-SSL"]),
        ("SimCLR-SSL", os.path.join(base_dir, "simclr", "evaluation_metrics.txt"), MODEL_COLORS["SimCLR-SSL"]),
        ("MoCo-SSL", os.path.join(base_dir, "moco", "evaluation_metrics.txt"), MODEL_COLORS["MoCo-SSL"]),
    ]


def build_output_paths(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    return {
        "comparison_file": comparison_file,
        "comparison_per_class_file": comparison_file.replace(".csv", "_per_class.csv"),
        "comparison_plot": os.path.join(output_dir, "model_comparison_plots.png"),
        "comparison_report": os.path.join(output_dir, "comparison_report.txt"),
    }


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
        cls_match = re.search(rf"^\s*{cls}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content, re.MULTILINE)
        if cls_match:
            metrics["class_metrics"][cls] = {
                "precision": float(cls_match.group(1)),
                "recall": float(cls_match.group(2)),
                "f1": float(cls_match.group(3)),
            }

    return metrics


def load_all_metrics(model_specs):
    loaded = []
    for model_name, metrics_path, color in model_specs:
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


def plot_comparison(overall_df, class_df, comparison_plot):
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
    plt.savefig(comparison_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(overall_df, class_df, comparison_report):
    best_idx = overall_df["Overall Accuracy"].idxmax()
    best_row = overall_df.loc[best_idx]

    baseline_row = overall_df[overall_df["Model"] == "Baseline (No SSL)"]
    baseline_acc = baseline_row["Overall Accuracy"].iloc[0] if not baseline_row.empty else np.nan

    with open(comparison_report, "w") as f:
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
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir or args.base_dir)
    model_specs = build_model_specs(base_dir)
    output_paths = build_output_paths(output_dir)

    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")

    loaded_metrics = load_all_metrics(model_specs)
    if len(loaded_metrics) < 2:
        raise RuntimeError("Need at least two evaluation_metrics.txt files to compare models.")

    overall_df, class_df = build_tables(loaded_metrics)
    overall_df.to_csv(output_paths["comparison_file"], index=False)
    class_df.to_csv(output_paths["comparison_per_class_file"], index=False)
    plot_comparison(overall_df, class_df, output_paths["comparison_plot"])
    write_report(overall_df, class_df, output_paths["comparison_report"])

    print(f"Saved comparison table: {output_paths['comparison_file']}")
    print(f"Saved comparison per-class table: {output_paths['comparison_per_class_file']}")
    print(f"Saved comparison plot: {output_paths['comparison_plot']}")
    print(f"Saved comparison report: {output_paths['comparison_report']}")


if __name__ == "__main__":
    main()
