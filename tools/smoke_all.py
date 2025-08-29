import os
import sys
import glob
import json
import csv
import random
import warnings
from typing import List, Tuple, Dict, Any, Optional, Iterable, Union
# Use non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")
# Make sure "tools" package is importable when running this file directly
ROOT = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Import the local code after sys.path is set
import transitbench as tb
from transitbench import recover

# Light, local helpers to avoid cross-file dependency surprises
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def write_json(obj: Dict[str, Any], path: str, **kwargs) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, **({"indent": 2} | kwargs))

def slugify(s: str) -> str:
    # safe filename slug
    keep = [c if c.isalnum() or c in ("-", "_", ".") else "-" for c in s]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-_.") or "lightcurve"

# So our console isn’t flooded by benign warnings in quick smoke runs
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=r"overflow encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r"divide by zero", category=RuntimeWarning)

# Import the local tools implementations
from tools.figures import load_robustness as fig_load_robustness
from tools.figures import plot_tpr_by_depth as fig_plot_tpr_by_depth
from tools.figures import plot_robustness as fig_plot_robustness
from tools.robustness import robustness_curve_for_file


def local_aggregate_runs(runs_root: str, paper_dir: str = None) -> Dict[str, Any]:
    """
    Very small, self-contained aggregator that scans runs/*/*.json (excluding robustness)
    and writes paper/records_all.csv, then computes a per-method TPR.
    """
    paper_dir = paper_dir or PAPER_DIR
    ensure_dir(paper_dir)
    # Collect JSONs that are not robustness artifacts
    paths = glob.glob(os.path.join(runs_root, "**", "*.json"), recursive=True)
    paths = [p for p in paths if os.path.sep + "robustness" + os.path.sep not in p]

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}  # method -> {'n': ..., 'tp': ...}

    for p in paths:
        try:
            with open(p, "r") as fh:
                obj = json.load(fh)
        except Exception:
            continue

        label = obj.get("label") or os.path.splitext(os.path.basename(p))[0]
        methods = obj.get("methods", {}) or {}
        # Tolerate older artifacts where "methods" is a list or malformed
        if isinstance(methods, list):
            # Try to coerce list of dicts like [{"method": "oot-replace", "records": [...]}, ...]
            coerced = {}
            for entry in methods:
                if isinstance(entry, dict) and "method" in entry and "records" in entry:
                    coerced[entry["method"]] = {"records": entry["records"]}
            methods = coerced
        if not isinstance(methods, dict):
            continue
        for m, md in methods.items():
            recs = md.get("records") or []
            for r in recs:
                depth    = r.get("depth") or r.get("depth_inj") or r.get("inj_depth")
                duration = r.get("duration") or r.get("dur") or r.get("inj_duration")
                period   = r.get("period") or r.get("P_inj") or r.get("inj_period")
                detected = bool(r.get("detected"))
                snr_inj  = r.get("snr_inj") or r.get("snr_injected") or r.get("SNR_injected")
                snr_top  = r.get("snr_top") or r.get("snr_detected") or r.get("SNR_detected")

                rows.append({
                    "label": label,
                    "method": m,
                    "depth": depth,
                    "duration": duration,
                    "period": period,
                    "detected": detected,
                    "snr_inj": snr_inj,
                    "snr_top": snr_top,
                })

                counts.setdefault(m, {"n": 0, "tp": 0})
                counts[m]["n"] += 1
                if detected:
                    counts[m]["tp"] += 1

    # Write CSV
    rec_csv = os.path.join(paper_dir, "records_all.csv")
    with open(rec_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["label", "method", "depth", "duration", "period", "detected", "snr_inj", "snr_top"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tpr_by_method = {
        m: {"n": v["n"], "tpr": (v["tp"] / v["n"]) if v["n"] else 0.0}
        for m, v in counts.items()
    }

    return {"counts": {"rows": len(rows)}, "tpr_by_method": tpr_by_method}


def local_load_robustness(runs_root: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Minimal robustness loader: scans runs/robustness/*.json and builds
    method -> [(epsilon, mean TPR across files), ...].
    """
    rb_dir = os.path.join(runs_root, "robustness")
    paths = glob.glob(os.path.join(rb_dir, "**", "*.json"), recursive=True)

    # method -> eps -> {'tp': int, 'n': int}
    acc: Dict[str, Dict[float, Dict[str, int]]] = {}

    for p in paths:
        try:
            with open(p, "r") as fh:
                obj = json.load(fh)
        except Exception:
            continue

        methods = obj.get("methods", {}) or {}
        for m, md in methods.items():
            recs = md.get("records") or []
            # Some writers may stash a constant eps for the method
            default_eps = md.get("adversarial_eps") or obj.get("adversarial_eps")

            for r in recs:
                eps = r.get("adversarial_eps")
                if eps is None:
                    eps = default_eps if default_eps is not None else 0.0
                try:
                    eps_f = float(eps)
                except Exception:
                    continue

                det = bool(r.get("detected"))
                acc.setdefault(m, {}).setdefault(eps_f, {"tp": 0, "n": 0})
                acc[m][eps_f]["n"] += 1
                if det:
                    acc[m][eps_f]["tp"] += 1

    curves: Dict[str, List[Tuple[float, float]]] = {}
    for m, by_eps in acc.items():
        pts: List[Tuple[float, float]] = []
        for eps, tn in sorted(by_eps.items()):
            n = tn["n"]
            tpr = (tn["tp"] / n) if n else 0.0
            pts.append((eps, tpr))
        if pts:
            curves[m] = pts
    return curves

from collections import defaultdict

def normalize_curves_data(curves: Any) -> Dict[str, List[Tuple[float, float]]]:
    """Coerce various "robustness curves" shapes into
    {method: [(eps, tpr), ...]}.

    Accepts:
      - dict: {method: [(eps,tpr), ...]}  (already OK)
      - dict: {method: {eps: tpr or {"tpr": ...}}}
      - list of dicts: each like {"method": "...", "points"|"curve"|..., ...}
      - list of point dicts: each like {"method": "...", "epsilon": ..., "tpr": ...}
    """
    if not curves:
        return {}

    # If already a mapping of method -> list/series, try to normalize each value.
    if isinstance(curves, dict):
        out: Dict[str, List[Tuple[float, float]]] = {}
        for m, pts in curves.items():
            # dict of eps -> (tpr or {tpr: ..})
            if isinstance(pts, dict):
                norm: List[Tuple[float, float]] = []
                for k, v in pts.items():
                    try:
                        eps = float(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        val = v.get("tpr") or v.get("TPR") or v.get("value")
                    else:
                        val = v
                    try:
                        tpr = float(val)
                    except Exception:
                        continue
                    norm.append((eps, tpr))
                if norm:
                    out[m] = sorted(norm, key=lambda x: x[0])
                continue

            # list/iterable of pairs or dicts
            norm = []
            try:
                iterator = list(pts)
            except Exception:
                iterator = []
            for p in iterator:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    e, t = p[0], p[1]
                elif isinstance(p, dict):
                    e = p.get("epsilon") or p.get("eps") or p.get("adversarial_eps")
                    t = p.get("tpr") or p.get("TPR") or p.get("value")
                else:
                    continue
                try:
                    e = float(e)
                    t = float(t)
                except Exception:
                    continue
                norm.append((e, t))
            if norm:
                out[m] = sorted(norm, key=lambda x: x[0])
        return out

    # If it's a list, try to merge entries per method
    if isinstance(curves, list):
        acc: Dict[str, List[Tuple[float, float]]] = {}
        # Case A: list of point dicts or list of method-buckets
        for entry in curves:
            if isinstance(entry, dict):
                m = entry.get("method") or entry.get("name") or entry.get("detector")
                pts = (
                    entry.get("points")
                    or entry.get("curve")
                    or entry.get("data")
                    or entry.get("pts")
                    or entry.get("values")
                    or entry.get("pairs")
                    or entry.get("eps_tpr")
                )
                # If not a bucket, maybe it's a single (epsilon,tpr) point.
                if not pts:
                    e = entry.get("epsilon") or entry.get("eps") or entry.get("adversarial_eps")
                    t = entry.get("tpr") or entry.get("TPR")
                    if m is None:
                        m = entry.get("method") or "method"
                    if e is not None and t is not None:
                        pts = [(e, t)]
                # Normalize pts
                norm = []
                if isinstance(pts, dict):
                    for k, v in pts.items():
                        try:
                            e = float(k)
                        except Exception:
                            continue
                        if isinstance(v, dict):
                            val = v.get("tpr") or v.get("TPR") or v.get("value")
                        else:
                            val = v
                        try:
                            t = float(val)
                        except Exception:
                            continue
                        norm.append((e, t))
                else:
                    try:
                        iterator = list(pts or [])
                    except Exception:
                        iterator = []
                    for p in iterator:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            e, t = p[0], p[1]
                        elif isinstance(p, dict):
                            e = p.get("epsilon") or p.get("eps") or p.get("adversarial_eps")
                            t = p.get("tpr") or p.get("TPR") or p.get("value")
                        else:
                            continue
                        try:
                            e = float(e)
                            t = float(t)
                        except Exception:
                            continue
                        norm.append((e, t))
                if norm:
                    acc.setdefault(m or "method", []).extend(norm)
        # Average duplicates by epsilon and sort
        final: Dict[str, List[Tuple[float, float]]] = {}
        for m, pts in acc.items():
            bucket: Dict[float, List[float]] = defaultdict(list)
            for e, t in pts:
                bucket[e].append(t)
            final[m] = sorted([(e, sum(v)/len(v)) for e, v in bucket.items()], key=lambda x: x[0])
        return final

    # Unknown shape
    return {}

# ---- paths ----
DATA_ROOT = os.path.join(ROOT, "data", "raw")
RUNS_ROOT = os.path.join(ROOT, "runs")
PAPER_DIR = os.path.join(ROOT, "paper")
FIG_DIR   = os.path.join(PAPER_DIR, "figs")

ensure_dir(RUNS_ROOT); ensure_dir(PAPER_DIR); ensure_dir(FIG_DIR)

# Pick 1–3 fast sample files (fallback to your known path)
FALLBACK = os.path.join(
    os.path.dirname(__file__),
    "sample_data",
    "false_positives",
    "9727392_236.01",
    "sector2",
    "hlsp_tess-data-alerts_tess_phot_00009727392-s02_tess_v1_lc.csv"
)

def pick_sample_paths(n: int = 12) -> List[str]:
    pats = [
        os.path.join(DATA_ROOT, "**", "*.csv"),
        os.path.join(DATA_ROOT, "**", "*.fits"),
        os.path.join(DATA_ROOT, "**", "*.tbl"),
    ]
    found: List[str] = []
    for p in pats:
        found.extend(glob.glob(p, recursive=True))
    found = sorted(set(found))  # de-dup

    def ok(path: str) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lc = tb.load(path)
            return getattr(lc, "n_pts", 0) >= 2000
        except Exception:
            return False

    found = [p for p in found if ok(p)]
    if not found:
        return [FALLBACK]
    random.seed(42)
    random.shuffle(found)
    return found[:min(n, len(found))]

# Minimal grid for speed (1×1×1 = 1 injection per method)
DEPTHS    = (0.003,0.005,0.010)     # 0.5%
DURATIONS = (0.08,0.12,0.20)      # 12% of period
PERIODS   = (1.5,3.0,5.0)       # 3 days

METHODS   = ("oot-replace", "prewhiten")
PROFILE   = "balanced"

def quick_benchmark(path: str, compute_budget: Optional[int] = 400) -> Dict[str, Any]:
    """Load one LC, run both methods once, save a result JSON under runs/smoke/."""
    print(f"[benchmark] {os.path.basename(path)}")
    lc = tb.load(path)
    print(f"  n_pts={lc.n_pts}")

    out: Dict[str, Any] = {
        "label": lc.label,
        "profile": PROFILE,
        "methods": {}
    }

    # Shared knobs; tiny grid; budgeted run
    shared = dict(
        per_injection_search=True,
        decision_metric="zscore",
        durations=list(DURATIONS),
        depths=list(DEPTHS),
        periods=list(PERIODS),
        compute_budget=compute_budget,
        adversarial=False,
    )

    for m in METHODS:
        res = lc.benchmark(method=(m,), profile=PROFILE, **shared)
        out["methods"][m] = res["methods"][m]
        recs = out["methods"][m].get("records", []) or []
        if not recs:
            print(f"  WARN: no records for {m}")
        else:
            r0 = recs[0]
            missing = [k for k in ("depth","duration","period","detected") if k not in r0]
            if missing:
                print(f"  WARN: {m} first record missing {missing}")

    # Save artifact
    out_dir  = ensure_dir(os.path.join(RUNS_ROOT, "smoke"))
    out_path = os.path.join(out_dir, f"{slugify(lc.label or os.path.basename(path))}.json")
    write_json(out, out_path, indent=2)
    print(f"  -> wrote {out_path}")
    return out

def quick_robustness(paths: Iterable[str]) -> None:
    """Tiny adversarial sweep that writes one aggregated JSON per file (eps = 0.0, 0.25, 0.5, 1.0)."""
    print("[robustness] running tiny sweep (eps=[0.0, 0.25, 0.5, 1.0])")
    for p in paths:
        try:
            _ = robustness_curve_for_file(
                p,
                methods=METHODS,
                profile=PROFILE,
                epsilons=(0.0, 0.25, 0.5, 1.0),
                n_injections=1,
                durations=DURATIONS,
                depths=DEPTHS,
                periods=PERIODS,
                per_injection_search=True,
                decision_metric="zscore",
                compute_budget=300,
            )
        except Exception as e:
            print(f"[robustness] {os.path.basename(p)} -> error: {e}")
    print("  -> robustness JSONs saved under runs/robustness/")

def quick_aggregate_and_figures() -> None:
    """Aggregate all runs and draw two simple figures."""
    print("[aggregate] collecting runs/*")
    rec_csv = os.path.join(PAPER_DIR, "records_all.csv")

    # Always use the local, schema-tolerant aggregator in this script
    summary = local_aggregate_runs(RUNS_ROOT, PAPER_DIR)
    print("  rows:", summary["counts"]["rows"])
    print("  TPR by method:", summary["tpr_by_method"])

    # Load robustness curves via tools.figures (expects one JSON per file with epsilons + methods)
    curves = fig_load_robustness(RUNS_ROOT)
    if curves:
        fig_plot_robustness(curves, FIG_DIR)
    else:
        print("  WARN: no robustness curves found (runs/robustness may be empty)")

    if os.path.exists(rec_csv):
        fig_plot_tpr_by_depth(rec_csv, FIG_DIR)
        print("  -> figs/tpr_by_depth_*.png")
    else:
        print(f"  WARN: {rec_csv} not found; skipping TPR-by-depth figure")

def main():
    n_files = int(os.environ.get("TB_SMOKE_N", "6"))
    paths = pick_sample_paths(n_files)
    if not paths:
        paths = [FALLBACK]
    print(f"[smoke] Using {len(paths)} file(s)")
    for p in paths:
        try:
            quick_benchmark(p)
        except Exception as e:
            print(f"  ERROR during benchmark on {os.path.basename(p)}: {type(e).__name__}: {e}")
    try:
        quick_robustness(paths[:2])
    except Exception as e:
        print(f"  ERROR during robustness: {e}")
    try:
        quick_aggregate_and_figures()
    except Exception as e:
        print(f"  ERROR during aggregate/figures: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

# /transitbench/src/transitbench/core.py
def benchmark(
    self,
    method: Union[str, Iterable[str]] = ("oot-replace",),
    *,
    compare_with: Optional[Iterable[str]] = None,
    profile: str = "balanced",
    # search / decision controls propagated to recover
    per_injection_search: bool = True,
    decision_metric: str = "zscore",
    snr_threshold: Optional[float] = None,
    rel_window: Optional[float] = None,     # fractional window around injected P
    durations: Optional[Iterable[float]] = (0.08, 0.12, 0.20),
    depths: Optional[Iterable[float]] = (0.003, 0.005, 0.010),
    periods: Optional[Iterable[float]] = (1.5, 3.0, 5.0),
    compute_budget: Optional[int] = None,
    cost_model: Optional["recover.CostModel"] = None,
    # detrending specifics
    oot_strategy: str = "median",
    prewhiten_nharm: int = 3,
    # adversarial / robustness knobs (forwarded; default off)
    adversarial: bool = False,
    adversarial_eps: float = 0.0,
    adversarial_seed: Optional[int] = None,
    **aliases,
) -> Dict[str, Any]:
    # Back-compat: accept old kw names used by some tools
    if "inj_durations" in aliases:
        durations = aliases.pop("inj_durations")
    if "inj_periods" in aliases:
        periods = aliases.pop("inj_periods")
    if "inj_depths" in aliases:
        depths = aliases.pop("inj_depths")