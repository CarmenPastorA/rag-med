"""Generate comparison plots from evaluation logs.

This script loads summary metrics from all JSON logs located in
tools/arqa/evaluation/*/logs directories and creates:
- A unified plot with 4 subplots comparing all retriever runs
- Individual bar plots per metric

The metrics plotted are normalized precision, recall, hit rate and mean reciprocal rank (MRR).
"""

import os
import sys
import json
import glob
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for consistency with other scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from shared import dunder_info

dunder_info.inject_dunder(__name__)

# Directories containing evaluation logs
FAISS_LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/faiss/logs")
BM25_LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/bm25/logs")
HYBRID_LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/hybrid/logs")

# Output directory for generated plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_log_metrics(log_file: str) -> Dict[str, str]:
    """Load summary metrics from a single evaluation log."""
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})

    # Force run_name to always be the json filename (without extension)
    run_name = os.path.splitext(os.path.basename(log_file))[0]

    # Determine retriever method from path or run_name
    if "bm25" in log_file or "bm25" in run_name.lower():
        method = "bm25"
    elif "hybrid" in log_file or "hybrid" in run_name.lower():
        method = "hybrid"
    elif "primary_faiss" in run_name.lower():
        method = "faiss"  # treat primary_faiss as faiss
    else:
        method = "faiss"

    return {
        "run_name": run_name,
        "method": method,
        "normalized_precision": summary.get("mean_normalized_precision@k", 0),
        "recall": summary.get("mean_recall@k", 0),
        "hit_rate": summary.get("mean_hit@k", 0),
        "mrr": summary.get("mean_mrr", 0),
    }



def collect_all_metrics(directories: List[str]) -> pd.DataFrame:
    """Read all JSON logs from the given directories and return a dataframe."""
    records: List[Dict[str, str]] = []
    for directory in directories:
        for log_file in glob.glob(os.path.join(directory, "*.json")):
            records.append(load_log_metrics(log_file))

    df = pd.DataFrame(records)
    return df


def plot_unified_metrics(df: pd.DataFrame, output_path: str) -> None:
    """Create a unified 4-subplot figure comparing all metrics with custom order."""
    # Color palette similar to example provided
    palette = {
        "bm25": "#5B8C5A",    # green tone
        "faiss": "#E69500",   # orange tone
        "hybrid": "#5B7DB1",  # blue tone
    }

    # Metrics to plot
    metrics = [
        ("normalized_precision", "Precisión Normalizada @k"),
        ("recall", "Recall @k"),
        ("hit_rate", "Hit Rate @k"),
        ("mrr", "MRR"),
    ]

    # Manual order
    order = [
        #"bm25_k10",
        "bm25_k50",
        #"faiss_k10",
        "faiss_k50",
        #"faiss_esencial_k10",
        "faiss_esencial_k50",
        #"latefusion_k10",
        "latefusion_k50",
        #"latefusion_k50_alpha0_2",
        #"latefusion_k50_alpha0_6",
        "hierarchical_faiss_k50",
        "late_fusion_fallback"
    ]

    # Apply order as categorical
    df["run_name"] = pd.Categorical(df["run_name"], categories=order, ordered=True)
    df = df[df["run_name"].isin(order)]
    df_sorted = df.sort_values("run_name")

    # Prepare labels and colors
    x_labels = df_sorted["run_name"].tolist()
    color_list = [palette.get(m, "gray") for m in df_sorted["method"]]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics):
        axes[i].bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=color_list,
            alpha=0.3,          # match style of your example
            edgecolor="black"
        )
        axes[i].set_title(title, fontsize=14)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(range(len(df_sorted)))
        axes[i].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_individual_metric(df: pd.DataFrame, metric: str, title: str, output_path: str) -> None:
    """Create a bar chart for a single metric and save to disk."""
    palette = {
        "bm25": "#5B8C5A",
        "faiss": "#E69500",
        "hybrid": "#5B7DB1",
    }

    # Manual order
    order = [
        #"bm25_k10",
        "bm25_k50",
        #"faiss_k10",
        "faiss_k50",
        #"faiss_esencial_k10",
        "faiss_esencial_k50",
        #"latefusion_k10",
        "latefusion_k50",
        #"latefusion_k50_alpha0_2",
        #"latefusion_k50_alpha0_6",
        "hierarchical_faiss_k50",
        "late_fusion_fallback"
    ]

    # Apply order as categorical
    df["run_name"] = pd.Categorical(df["run_name"], categories=order, ordered=True)
    df = df[df["run_name"].isin(order)]
    df_sorted = df.sort_values("run_name")

    x_labels = df_sorted["run_name"].tolist()
    color_list = [palette.get(m, "gray") for m in df_sorted["method"]]

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(df_sorted)),
        df_sorted[metric],
        color=color_list,
        alpha=0.3,
        edgecolor="black"
    )
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(range(len(df_sorted)), x_labels, rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



if __name__ == "__main__":
    log_dirs = [FAISS_LOG_DIR, BM25_LOG_DIR, HYBRID_LOG_DIR]
    df_metrics = collect_all_metrics(log_dirs)

    # Generate unified plot
    unified_out_file = os.path.join(PLOTS_DIR, "summary_comparison.png")
    plot_unified_metrics(df_metrics, unified_out_file)

    # Generate individual metric plots
    metrics_to_plot = [
        ("normalized_precision", "Precisión Normalizada @k"),
        ("recall", "Recall @k"),
        ("hit_rate", "Hit Rate @k"),
        ("mrr", "MRR"),
    ]

    #for metric, title in metrics_to_plot:
    #    out_file = os.path.join(PLOTS_DIR, f"{metric}_comparison.png")
    #    plot_individual_metric(df_metrics, metric, title, out_file)

    print(f"Unified plot and individual plots saved to: {PLOTS_DIR}")

