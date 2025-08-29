# /Users/eshaantripathi/Documents/transitbench/tools/figures.py
from __future__ import annotations
import os, json, glob
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from transitbench.utils import ensure_dir

def load_robustness(runs_root: str) -> List[dict]:
    paths = glob.glob(os.path.join(runs_root, "robustness", "*.json"))
    out = []
    for p in paths:
        try:
            with open(p, "r") as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out

def plot_robustness(curves: List[dict], out_dir: str):
    ensure_dir(out_dir)
    # aggregate by method (average TPR over files)
    methods = sorted({m for c in curves for m in c.get("methods", {}).keys()})
    if not methods or not curves: return None

    eps = curves[0]["epsilons"]
    for m in methods:
        mat = []
        for c in curves:
            if m in c.get("methods", {}):
                mat.append(c["methods"][m]["tpr"])
        if not mat: continue
        arr = np.array(mat)  # (n_files, n_eps)
        mean = np.nanmean(arr, axis=0)
        p10  = np.nanpercentile(arr, 10, axis=0)
        p90  = np.nanpercentile(arr, 90, axis=0)

        plt.figure()
        plt.plot(eps, mean, label=f"{m} mean")
        plt.fill_between(eps, p10, p90, alpha=0.2, label="10–90%")
        plt.xlabel("Adversarial ε")
        plt.ylabel("TPR")
        plt.title(f"Robustness curve — {m}")
        plt.legend()
        fig_path = os.path.join(out_dir, f"robustness_{m}.png")
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close()

def plot_tpr_by_depth(records_csv: str, out_dir: str):
    """
    Expects the CSV from aggregate_runs. We'll compute TPR by depth per method.
    """
    import pandas as pd
    ensure_dir(out_dir)
    df = pd.read_csv(records_csv)
    if "detected" not in df.columns or "depth" not in df.columns or "method" not in df.columns:
        return
    df = df.dropna(subset=["depth"])
    # group
    methods = sorted(df["method"].unique())
    depths = sorted(df["depth"].unique())
    for m in methods:
        sub = df[df["method"] == m]
        tprs, ds = [], []
        for d in depths:
            ss = sub[np.isclose(sub["depth"], d)]
            if len(ss) == 0:
                continue
            tprs.append(ss["detected"].astype(bool).mean())
            ds.append(d)
        if not tprs: continue
        plt.figure()
        plt.plot(ds, tprs, marker="o")
        plt.xlabel("Injected depth")
        plt.ylabel("TPR")
        plt.title(f"TPR vs depth — {m}")
        fig_path = os.path.join(out_dir, f"tpr_by_depth_{m}.png")
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close()

# Example programmatic use:
if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(__file__), "..", "runs")
    curves = load_robustness(ROOT)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
    plot_robustness(curves, out_dir)
    # If you've already run aggregate_runs and have paper/records_all.csv:
    rec_csv = os.path.join(os.path.dirname(__file__), "..", "paper", "records_all.csv")
    if os.path.exists(rec_csv):
        plot_tpr_by_depth(rec_csv, out_dir)