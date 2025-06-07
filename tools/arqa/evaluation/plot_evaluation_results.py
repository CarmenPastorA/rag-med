"""Generate comparison plots from evaluation logs.

This script loads summary metrics from all JSON logs located in
tools/arqa/evaluation/*/logs directories and creates bar plots to
compare different retriever runs. The metrics plotted are normalized
precision, recall, hit rate and mean reciprocal rank (MRR).
"""

import os
import json
import glob
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for consistency with other scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

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
    """Load summary metrics from a single evaluation log.

    Parameters
    ----------
    log_file : str
        Path to the JSON log file.

    Returns
    -------
    Dict[str, str]
        Dictionary with the run name, retriever type and selected metrics.
    """
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    run_name = summary.get("run_name", os.path.splitext(os.path.basename(log_file))[0])

    # Determine retriever method from path
    if "bm25" in log_file:
        method = "bm25"
    elif "hybrid" in log_file:
        method = "hybrid"
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
    """Read all JSON logs from the given directories and return a dataframe.

    Parameters
    ----------
    directories : List[str]
        List of directories that contain evaluation logs.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per evaluation run and columns for each metric.
    """
    records: List[Dict[str, str]] = []
    for directory in directories:
        for log_file in glob.glob(os.path.join(directory, "*.json")):
            records.append(load_log_metrics(log_file))

    df = pd.DataFrame(records)
    return df


def plot_metric(df: pd.DataFrame, metric: str, output_path: str) -> None:
    """Create a bar chart for a single metric and save to disk."""
    colors = {"bm25": "tab:blue", "faiss": "tab:orange", "hybrid": "tab:green"}

    df_sorted = df.sort_values("method")
    x_pos = range(len(df_sorted))
    color_list = [colors.get(m, "gray") for m in df_sorted["method"]]

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, df_sorted[metric], color=color_list)
    plt.xticks(x_pos, df_sorted["run_name"], rotation=45, ha="right", fontsize=8)
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    log_dirs = [FAISS_LOG_DIR, BM25_LOG_DIR, HYBRID_LOG_DIR]
    df_metrics = collect_all_metrics(log_dirs)

    for metric in ["normalized_precision", "recall", "hit_rate", "mrr"]:
        out_file = os.path.join(PLOTS_DIR, f"{metric}_comparison.png")
        plot_metric(df_metrics, metric, out_file)

    print(f"Plots saved to: {PLOTS_DIR}")
