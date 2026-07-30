# src/eval/visualization.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def draw_annotations(img, tps, fps, fns, is_cowc=False):
    canvas = img.copy()
    for fn in fns:
        if is_cowc: cv2.circle(canvas, (int(fn["gt_pt"][0]), int(fn["gt_pt"][1])), 6, (255, 0, 0), -1)
        else: cv2.polylines(canvas, [np.array(fn["gt_box"], dtype=np.int32).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
    for fp in fps: cv2.polylines(canvas, [np.array(fp["pred_box"], dtype=np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)
    for tp in tps: cv2.polylines(canvas, [np.array(tp["pred_box"], dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    return canvas

def generate_powerpoint_plots(df, all_preds, output_dir, model_name):
    plots_dir = output_dir / "ppt_plots"
    plots_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 1. Grouped Bar Chart - mAP50
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Dataset", y="mAP50", hue="Variant", palette="viridis")
    plt.title(f"[{model_name}] mAP@50 Across Resolution Variants", fontweight="bold")
    plt.ylim(0, 1.05); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(plots_dir / "bar_map50_grouped.png", dpi=300); plt.close()

    # 2. Grouped Bar Chart - F1 Score
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Dataset", y="F1_Score", hue="Variant", palette="mako")
    plt.title(f"[{model_name}] F1 Score Across Resolution Variants", fontweight="bold")
    plt.ylim(0, 1.05); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(plots_dir / "bar_f1_grouped.png", dpi=300); plt.close()

    # 3. ROC Curves
    for ds in df["Dataset"].unique():
        plt.figure(figsize=(9, 7))
        ds_preds = [p for p in all_preds if p["dataset"] == ds]
        for var in sorted(list(set([p["variant"] for p in ds_preds]))):
            var_preds = [p for p in ds_preds if p["variant"] == var]
            if not var_preds: continue
            y_true, y_scores = [1 if p["is_tp"] else 0 for p in var_preds], [p["score"] for p in var_preds]
            if len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                plt.plot(fpr, tpr, lw=2, label=f"{var} (AUC = {auc(fpr, tpr):.2f})")
        
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
        plt.title(f"[{model_name}] ROC Curve - {ds}", fontweight="bold")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(plots_dir / f"roc_curve_{ds}.png", dpi=300); plt.close()

    # Data outputs
    df_clean = df[["Dataset", "Variant", "GSD_cm", "DegradationType", "TP", "FP", "FN", "Precision", "Recall", "F1_Score", "mAP50", "Mean_IoU", "Mean_Point_Dist_px"]]
    
    try: tex_output = df_clean.to_latex(index=False, caption=f"Metrics for {model_name}.", label="tab:rq1_results", column_format="llcrrrrrrrrrr")
    except AttributeError: tex_output = df_clean.style.to_latex(caption=f"Metrics for {model_name}.", label="tab:rq1_results")
        
    with open(output_dir / "rq1_paper_table.tex", "w") as f: f.write(tex_output)
    with open(output_dir / "RQ1_ANSWER_SUMMARY.md", "w") as f: f.write(f"# RQ1 Summary\n**Model:** `{model_name}`\n\n{df_clean.to_markdown(index=False)}")

def generate_extended_plots(df, vedai_iou_data, vedai_size_data, output_dir, model_name):
    plots_dir = output_dir / "ppt_plots"
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 1. Degradation Curve
    for ds in df["Dataset"].unique():
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df[df["Dataset"] == ds].sort_values("GSD_cm"), x="GSD_cm", y="mAP50", hue="DegradationType", marker="o", lw=2.5)
        plt.title(f"[{model_name}] {ds} - Degradation Curve", fontweight="bold")
        plt.ylim(0, 1.05); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
        plt.savefig(plots_dir / f"degradation_curve_{ds}.png", dpi=300); plt.close()

    # 2. VEDAI IoU Distribution
    if vedai_iou_data:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=pd.DataFrame(vedai_iou_data), x="iou", hue="variant", common_norm=False, fill=True, alpha=0.3, linewidth=2)
        plt.title(f"[{model_name}] VEDAI - True Positive IoU Distribution", fontweight="bold")
        plt.xlim(0.4, 1.0); plt.tight_layout()
        plt.savefig(plots_dir / "vedai_iou_distribution.png", dpi=300); plt.close()

    # 3. VEDAI Size Stratified Recall
    if vedai_size_data:
        sz_df = pd.DataFrame(vedai_size_data)
        sz_df = sz_df[sz_df["area"] > 0]
        if not sz_df.empty:
            try: sz_df['Size_Bin'] = pd.qcut(sz_df['area'], q=3, labels=['Small', 'Medium', 'Large'], duplicates='drop')
            except ValueError: sz_df['Size_Bin'] = pd.cut(sz_df['area'], bins=3, labels=['Small', 'Medium', 'Large'])
            
            recall_df = sz_df.groupby(['variant', 'Size_Bin'])['matched'].mean().reset_index().rename(columns={'matched': 'Recall'})
            plt.figure(figsize=(12, 6))
            sns.barplot(data=recall_df, x="Size_Bin", y="Recall", hue="variant", palette="Set2")
            plt.title(f"[{model_name}] VEDAI - Size Stratified Recall", fontweight="bold")
            plt.ylim(0, 1.05); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
            plt.savefig(plots_dir / "vedai_size_stratified_recall.png", dpi=300); plt.close()