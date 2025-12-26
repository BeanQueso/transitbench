from __future__ import annotations
import numpy as np
import csv
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def _coverage_bin_label(x: float) -> str:
    if x < 0.2: return "0.0–0.2"
    if x < 0.5: return "0.2–0.5"
    if x < 0.8: return "0.5–0.8"
    return "0.8–1.0"

def _bar(values, labels, ylabel, title, out_path):
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_detection_vs_coverage(records: List[Dict], out_path: str):
    bins = {"0.0–0.2": [], "0.2–0.5": [], "0.5–0.8": [], "0.8–1.0": []}
    for r in records:
        cov = r.get("phase_coverage", np.nan)
        if not np.isfinite(cov): continue
        bins[_coverage_bin_label(float(cov))].append(r)
    labels = list(bins.keys())
    fracs = []
    for k in labels:
        grp = bins[k]
        fracs.append(0.0 if len(grp)==0 else 100.0 * sum(int(x["detected"]) for x in grp) / len(grp))
    _bar(fracs, labels, "Detection fraction (%)", "Detection vs Phase Coverage", out_path)

def plot_detection_vs_events(records: List[Dict], out_path: str):
    groups = {"0": [], "1": [], "2": [], "≥3": []}
    for r in records:
        k = r.get("n_transits_observed", 0)
        lab = "≥3" if (isinstance(k, (int,float)) and k >= 3) else str(int(k))
        if lab in groups: groups[lab].append(r)
    labels = ["0","1","2","≥3"]
    fracs = []
    for k in labels:
        grp = groups[k]
        fracs.append(0.0 if len(grp)==0 else 100.0 * sum(int(x["detected"]) for x in grp) / len(grp))
    _bar(fracs, labels, "Detection fraction (%)", "Detection vs # Observed Events", out_path)

def plot_completeness_heatmap(records: List[Dict], out_path: str):
    """Heatmap of detection fraction vs (depth, period), aggregated over durations."""
    # collect unique sorted depths/periods as they appear in records
    depths = sorted({float(r["depth"]) for r in records})
    periods = sorted({float(r["period"]) for r in records})
    di = {d:i for i,d in enumerate(depths)}
    pi = {p:i for i,p in enumerate(periods)}
    num = np.zeros((len(depths), len(periods)), dtype=float)
    den = np.zeros_like(num)
    for r in records:
        i = di[float(r["depth"])]; j = pi[float(r["period"])]
        den[i,j] += 1.0
        if r.get("detected", False): num[i,j] += 1.0
    frac = np.divide(num, den, out=np.zeros_like(num), where=den>0)
    plt.figure()
    extent = (min(periods), max(periods), min(depths), max(depths))
    im = plt.imshow(frac, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Period (days)")
    plt.ylabel("Depth")
    cbar = plt.colorbar(im)
    cbar.set_label("Detection fraction")
    plt.title("Completeness (aggregated over durations)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def write_records_csv(records: List[Dict], out_path: str):
    if not records: 
        # still write header for consistency
        fieldnames = ["depth","duration","period","detected","snr_top","snr_injected",
                      "period_rec","t0_injected_used","phase_coverage","n_transits_observed"]
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    fieldnames = list({k for r in records for k in r.keys()})
    # stable order
    preferred = ["depth","duration","period","detected","snr_top","snr_injected",
                 "period_rec","t0_injected_used","phase_coverage","n_transits_observed"]
    fieldnames = preferred + [k for k in fieldnames if k not in preferred]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

def write_completeness_table_csv(records: List[Dict], out_path: str):
    """CSV of detection fraction per (depth, period), aggregated over durations."""
    depths = sorted({float(r["depth"]) for r in records})
    periods = sorted({float(r["period"]) for r in records})
    di = {d:i for i,d in enumerate(depths)}
    pi = {p:i for i,p in enumerate(periods)}
    num = np.zeros((len(depths), len(periods)), dtype=float)
    den = np.zeros_like(num)
    for r in records:
        i = di[float(r["depth"])]; j = pi[float(r["period"])]
        den[i,j] += 1.0
        if r.get("detected", False): num[i,j] += 1.0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth\\period"] + periods)
        for i,d in enumerate(depths):
            row = [d]
            for j,_ in enumerate(periods):
                frac = num[i,j]/den[i,j] if den[i,j]>0 else np.nan
                row.append(float(frac))
            w.writerow(row)
