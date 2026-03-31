"""Generate Phase 2, Phase 3, and final comparison plots from stage-comparison JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CLASS_NAMES = ["NC", "G3", "G5", "G4"]
MODEL_COLORS = {
    "Baseline (No SSL)": "#ff9999",
    "CAE-SSL": "#66b3ff",
    "SimCLR-SSL": "#99ff99",
    "MoCo-SSL": "#ffcc80",
}
PREFERRED_STAGE = {
    "Baseline (No SSL)": "Single Stage",
    "CAE-SSL": "Stage 2 (End-to-End)",
    "SimCLR-SSL": "Stage 2 (End-to-End)",
    "MoCo-SSL": "Stage 2 (End-to-End) [Selected]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison PNGs for Phase 2/3/final results.")
    parser.add_argument("--project-root", default=".", help="Path to project root")
    return parser.parse_args()


def normalize_records(records: list[dict], selection: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = selection or PREFERRED_STAGE
    overall_rows: list[dict] = []
    class_rows: list[dict] = []

    for model_name, desired_stage in selection.items():
        match = next((row for row in records if row["model"] == model_name and row["stage"] == desired_stage), None)
        if match is None:
            raise ValueError(f"Could not find record for {model_name} / {desired_stage}")

        overall_rows.append(
            {
                "Model": model_name,
                "Overall Accuracy": match["accuracy"],
                "Macro F1-Score": match["macro_f1"],
                "Weighted F1-Score": match["weighted_f1"],
                "Cohen's Kappa": match["kappa"],
                "Color": MODEL_COLORS[model_name],
            }
        )

        for class_name in CLASS_NAMES:
            per_class = match["per_class"][class_name]
            class_rows.append(
                {
                    "Model": model_name,
                    "Class": class_name,
                    "Precision": per_class["precision"],
                    "Recall": per_class["recall"],
                    "F1-Score": per_class["f1"],
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def plot_comparison(overall_df: pd.DataFrame, class_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(16, 12))
    colors = overall_df["Color"].tolist()
    labels = overall_df["Model"].tolist()

    fig.suptitle(title, fontsize=18, y=0.98)

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
    x = range(len(labels))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], overall_df["Macro F1-Score"], width, label="Macro F1", color="#ef9a9a", edgecolor="black")
    ax2.bar([i + width / 2 for i in x], overall_df["Weighted F1-Score"], width, label="Weighted F1", color="#90caf9", edgecolor="black")
    ax2.set_title("F1 Scores")
    ax2.set_xticks(list(x))
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

    for idx, class_name in enumerate(CLASS_NAMES):
        ax = plt.subplot(3, 3, 4 + idx)
        cls_data = class_df[class_df["Class"] == class_name]
        bar_colors = [MODEL_COLORS[model] for model in cls_data["Model"].tolist()]
        bars = ax.bar(range(len(cls_data)), cls_data["Recall"], color=bar_colors, edgecolor="black")
        ax.set_title(f"{class_name} Recall")
        ax.set_xticks(range(len(cls_data)))
        ax.set_xticklabels(cls_data["Model"].tolist(), rotation=10, ha="right")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tables(overall_df: pd.DataFrame, class_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_df.drop(columns=["Color"]).to_csv(output_dir / "model_comparison.csv", index=False)
    class_df.to_csv(output_dir / "model_comparison_per_class.csv", index=False)


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    phase2_records = load_records(project_root / "phase2_comparison" / "stage_comparison_detailed_results.json")
    phase3_records = load_records(project_root / "phase3_comparison" / "stage_comparison_detailed_results.json")

    phase2_overall, phase2_class = normalize_records(phase2_records)
    save_tables(phase2_overall, phase2_class, project_root / "phase2_comparison")
    plot_comparison(
        phase2_overall,
        phase2_class,
        project_root / "phase2_comparison" / "model_comparison_plots.png",
        "Phase 2 Comparison (Best Checkpoints on Common Test Set)",
    )

    phase3_overall, phase3_class = normalize_records(phase3_records)
    save_tables(phase3_overall, phase3_class, project_root / "phase3_comparison")
    plot_comparison(
        phase3_overall,
        phase3_class,
        project_root / "phase3_comparison" / "model_comparison_plots.png",
        "Phase 3 Comparison (Class-Weighted Fine-Tuning)",
    )

    final_selection = {
        "Baseline (No SSL)": "Single Stage",
        "CAE-SSL": "Stage 2 (End-to-End)",
        "SimCLR-SSL": "Stage 2 (End-to-End)",
        "MoCo-SSL": "Stage 2 (End-to-End) [Selected]",
    }
    final_records = []
    for record in phase2_records:
        if record["model"] in {"Baseline (No SSL)", "CAE-SSL", "SimCLR-SSL"} and record["stage"] == final_selection[record["model"]]:
            final_records.append(record)
    for record in phase3_records:
        if record["model"] == "MoCo-SSL" and record["stage"] == final_selection["MoCo-SSL"]:
            final_records.append(record)

    final_overall, final_class = normalize_records(final_records, selection=final_selection)
    final_dir = project_root / "final_comparison"
    save_tables(final_overall, final_class, final_dir)
    plot_comparison(
        final_overall,
        final_class,
        final_dir / "model_comparison_plots.png",
        "Final Locked Comparison (Phase 2 CAE/SimCLR, Phase 3 MoCo)",
    )

    print("Generated:")
    print(project_root / "phase2_comparison" / "model_comparison_plots.png")
    print(project_root / "phase3_comparison" / "model_comparison_plots.png")
    print(final_dir / "model_comparison_plots.png")


if __name__ == "__main__":
    main()
