import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Comparative PowerPoint Plots (YOLO vs RT-DETR)")
    parser.add_argument("--yolo-csv", type=str, required=True, help="Path to YOLO rq1_summary_metrics.csv")
    parser.add_argument("--detr-csv", type=str, required=True, help="Path to RT-DETR rq1_summary_metrics.csv")
    parser.add_argument("--yolo-name", type=str, default="YOLOv8n", help="Display name for YOLO model")
    parser.add_argument("--detr-name", type=str, default="RT-DETR", help="Display name for RT-DETR model")
    parser.add_argument("--out-dir", type=str, default="Comparison_Plots", help="Output directory for generated plots")
    return parser.parse_args()

def generate_comparative_plots(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set formatting for PowerPoint (Large, clear fonts, white background)
    sns.set_theme(style="whitegrid", font_scale=1.4)
    palette = "Set1" # Distinct contrasting colors for the two models

    datasets = df["Dataset"].unique()

    for ds in datasets:
        ds_df = df[df["Dataset"] == ds].sort_values(by=["GSD_cm", "Variant"])

        # ---------------------------------------------------------
        # 1. Comparative Degradation Curve (RQ1 / RQ2)
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 7))
        # Use hue for Degradation Type and style/markers for the Model Architecture
        sns.lineplot(
            data=ds_df, x="GSD_cm", y="mAP50", 
            hue="DegradationType", style="Model", 
            markers=True, dashes=True, linewidth=3, markersize=12, palette="Dark2"
        )
        plt.title(f"Architecture Degradation Comparison: {ds}", fontweight="bold", pad=15)
        plt.ylabel("mAP @ 0.50", fontweight="bold")
        plt.xlabel("Ground Sample Distance (GSD) in cm", fontweight="bold")
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_degradation_{ds}.png", dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 2. Head-to-Head mAP50 Bar Chart (RQ2)
        # ---------------------------------------------------------
        plt.figure(figsize=(14, 7))
        sns.barplot(data=ds_df, x="Variant", y="mAP50", hue="Model", palette=palette)
        plt.title(f"Head-to-Head mAP50 Performance: {ds}", fontweight="bold", pad=15)
        plt.ylabel("mAP @ 0.50", fontweight="bold")
        plt.xlabel("Dataset Variant", fontweight="bold")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_mAP50_bars_{ds}.png", dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 3. Head-to-Head Peak F1 Bar Chart (RQ2)
        # ---------------------------------------------------------
        plt.figure(figsize=(14, 7))
        sns.barplot(data=ds_df, x="Variant", y="F1_Score", hue="Model", palette=palette)
        plt.title(f"Head-to-Head Peak F1 Score: {ds}", fontweight="bold", pad=15)
        plt.ylabel("Peak F1 Score", fontweight="bold")
        plt.xlabel("Dataset Variant", fontweight="bold")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_F1_bars_{ds}.png", dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 4. False Positive Robustness Comparison (RQ4 implicitly)
        # ---------------------------------------------------------
        plt.figure(figsize=(14, 7))
        sns.barplot(data=ds_df, x="Variant", y="FP", hue="Model", palette="magma")
        plt.title(f"False Positive (FP) Counts at Peak Confidence: {ds}", fontweight="bold", pad=15)
        plt.ylabel("Total False Positives", fontweight="bold")
        plt.xlabel("Dataset Variant", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_FP_counts_{ds}.png", dpi=300)
        plt.close()

def main():
    args = parse_args()

    yolo_path = Path(args.yolo_csv)
    detr_path = Path(args.detr_csv)

    if not yolo_path.exists():
        print(f"Error: Could not find YOLO CSV at {yolo_path}")
        return
    if not detr_path.exists():
        print(f"Error: Could not find DETR CSV at {detr_path}")
        return

    # Load CSVs
    yolo_df = pd.read_csv(yolo_path)
    detr_df = pd.read_csv(detr_path)

    # Tag dataframes with model architecture
    yolo_df["Model"] = args.yolo_name
    detr_df["Model"] = args.detr_name

    # Merge into a single dataframe
    combined_df = pd.concat([yolo_df, detr_df], ignore_index=True)

    print(f"Generating Comparative Plots in: {args.out_dir}...")
    generate_comparative_plots(combined_df, args.out_dir)
    print("Success! PowerPoint comparative plots are ready.")

if __name__ == "__main__":
    main()