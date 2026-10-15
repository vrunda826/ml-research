"""Generate comprehensive figure for B0-B10 experimental suite comparison."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "reports" / "final_evaluation"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

def generate_b0_b10_comparison_plot():
    results_path = OUTPUT_DIR / "results_all_models.csv"
    df = pd.read_csv(results_path)
    
    # Filter unique canonical models in order B0 -> B10
    model_order = [
        "B0_Majority",
        "B1a_Audio_eGeMAPS",
        "B1b_Audio_MFCC",
        "B2_Text_MiniLM",
        "B3_Multimodal_MFCC_Text",
        "B4_Multitask_Net",
        "B5_Generic_SupCon",
        "B6_Ordinary_Oversampling",
        "B7_Severity_Controlled_Sampling",
        "B8_Hard_Negatives",
        "B9_DA_SupCon",
        "B10_Full_Proposed_Model",
    ]
    
    clean_labels = [
        "B0 (Majority)",
        "B1a (eGeMAPS)",
        "B1b (MFCC)",
        "B2 (Text)",
        "B3 (MM Base)",
        "B4 (Multitask)",
        "B5 (SupCon)",
        "B6 (Oversamp)",
        "B7 (DA-Samp)",
        "B8 (HardNeg)",
        "B9 (DA-SupCon)",
        "B10 (MDA-Net)",
    ]
    
    sub = df.set_index("model_id").loc[model_order].reset_index()
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.32)
    
    x = np.arange(len(model_order))
    width = 0.38
    
    # -------------------------------------------------------------
    # Panel 1: Primary Binary Classification (Macro-F1 & Recall)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    rects1 = ax1.bar(x - width/2, sub["macro_f1"], width, label="Macro-F1", color="#3182bd", edgecolor="k", alpha=0.85)
    rects2 = ax1.bar(x + width/2, sub["depression_recall"], width, label="Depression Recall", color="#de2d26", edgecolor="k", alpha=0.85)
    
    ax1.axhline(0.4286, color="gray", linestyle="--", alpha=0.7, label="Majority Floor (F1=0.43)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=9, fontweight="semibold")
    ax1.set_ylabel("Metric Score (0.0 to 1.0)")
    ax1.set_title("A. Binary Depression Detection Performance (Macro-F1 & Recall)\n(B10 achieves peak clinical recall = 72.7%)", fontweight="bold")
    ax1.set_ylim([0, 0.95])
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)
    
    # -------------------------------------------------------------
    # Panel 2: Continuous Regression Accuracy (MAE & R2)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_r2 = ax2.twinx()
    
    rects3 = ax2.bar(x - width/2, sub["mae"], width, label="MAE (lower is better)", color="#41b6c4", edgecolor="k", alpha=0.85)
    rects4 = ax2_r2.bar(x + width/2, sub["r2"], width, label="R² (higher is better)", color="#fe9929", edgecolor="k", alpha=0.85)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=9, fontweight="semibold")
    ax2.set_ylabel("Dev MAE (PHQ-8 points)", color="#081d58", fontweight="bold")
    ax2_r2.set_ylabel("Dev R² Score", color="#8c2d04", fontweight="bold")
    ax2.set_title("B. Continuous PHQ-8 Severity Estimation (MAE & R²)\n(B3 achieves lowest MAE=3.80; B10 achieves balanced MAE=3.92)", fontweight="bold")
    ax2.set_ylim([0, 5.5])
    ax2_r2.set_ylim([-0.25, 0.35])
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8.5)
    
    # -------------------------------------------------------------
    # Panel 3: High-PHQ Underprediction Discrepancy
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    colors = ["#7f7f7f", "#e66101", "#fdb863", "#5e3c99", "#0571b0", "#8073ac", "#b2abd2", "#fdb863", "#2b83ba", "#dfc27d", "#80cdc1", "#b2182b"]
    bars = ax3.bar(x, sub["dep_mean_discrepancy"], color=colors, edgecolor="k", alpha=0.85)
    
    ax3.axhline(8.2553, color="black", linestyle="--", linewidth=1.5, label="Dummy Regressor Reference (+8.26)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=9, fontweight="semibold")
    ax3.set_ylabel("Depressed Mean Discrepancy D_dep (PHQ-8 points)")
    ax3.set_title("C. High-PHQ Underprediction Gap across Models\n(Notice acoustic B1a matches Dummy; B3 and B10 significantly reduce gap)", fontweight="bold")
    ax3.set_ylim([0, 9.8])
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True)
    
    # -------------------------------------------------------------
    # Panel 4: Clinical Pareto Trade-off: Recall vs MAE
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    scatter = ax4.scatter(sub["mae"], sub["depression_recall"], s=sub["macro_f1"]*200, c=sub["r2"], cmap="plasma", edgecolors="k", linewidth=1.2, alpha=0.9)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Continuous R² Score")
    
    for i, txt in enumerate(clean_labels):
        offset_x = 0.04
        offset_y = 0.02
        if "B10" in txt:
            offset_y = -0.04
        ax4.annotate(txt, (sub["mae"].iloc[i] + offset_x, sub["depression_recall"].iloc[i] + offset_y), fontsize=8.5, fontweight="bold")
        
    ax4.set_xlabel("Continuous MAE (lower is better)")
    ax4.set_ylabel("Depression Recall / Sensitivity (higher is better)")
    ax4.set_title("D. Clinical Pareto Frontier: Recall vs. Continuous MAE\n(Bubble size = Macro-F1; B10 dominates high-recall frontier)", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.3)
    
    fig.suptitle("COMPLETE B0-B10 EXPERIMENTAL PROGRESSION FOR HSED DEPRESSION RESEARCH", fontsize=15, fontweight="bold", y=0.995)
    
    out_file = PLOTS_DIR / "plot_10_b0_b10_comprehensive_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    generate_b0_b10_comparison_plot()
