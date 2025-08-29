# /transitbench/tools/figures.py
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
    """
    Accepts a list of JSON payloads from either:
      (A) robustness_curve_for_file: {"epsilons": [...], "methods": {m:{"tpr":[...]}}}
      (B) run_robustness_for_list per-file JSONs: {"method": m, "epsilon": e, "result": {"records": [...]}}

    Produces mean curve with 10–90% band per method across files.
    """
    ensure_dir(out_dir)
    if not curves:
        print("WARN: no robustness curves found (nothing to plot)")
        return None

    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # method -> eps -> list[tpr]
    agg = defaultdict(lambda: defaultdict(list))

    for c in curves:
        if isinstance(c, dict) and "methods" in c and "epsilons" in c:
            eps = [float(e) for e in c.get("epsilons", [])]
            for m, d in c.get("methods", {}).items():
                tprs = d.get("tpr") or d.get("tprs")
                if not tprs:
                    continue
                for e, t in zip(eps, tprs):
                    try:
                        agg[m][float(e)].append(float(t))
                    except Exception:
                        pass
        elif isinstance(c, dict) and "method" in c and "epsilon" in c and "result" in c:
            m = c.get("method")
            e = float(c.get("epsilon"))
            recs = c.get("result", {}).get("records", [])
            if recs:
                tpr = float(np.mean([1 if r.get("detected") else 0 for r in recs]))
                agg[m][e].append(tpr)

    if not agg:
        print("WARN: robustness curves normalization produced no points")
        return None

    for m, series in agg.items():
        eps_sorted = sorted(series.keys())
        mat = np.array([[v for v in series[e]] for e in eps_sorted], dtype=float)  # (n_eps, n_files)
        mean = np.nanmean(mat, axis=1)
        p10  = np.nanpercentile(mat, 10, axis=1)
        p90  = np.nanpercentile(mat, 90, axis=1)

        plt.figure()
        plt.plot(eps_sorted, mean, marker="o", label=f"{m} mean")
        plt.fill_between(eps_sorted, p10, p90, alpha=0.2, label="10–90%")
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