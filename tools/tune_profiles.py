#!/usr/bin/env python3
# tools/tune_profiles.py
"""
Tune TransitBench detection profiles (sensitive / balanced / strict) from *null* trials.

Run with no arguments:
    python tools/tune_profiles.py

It will try to auto-discover light-curve files under ./data (and a few common subdirs),
auto-detect reasonable time/flux columns per file, run a batch of null trials
(no injection) for each method, and propose z-score thresholds (tau_base) that
achieve each profile's target false-positive rate.

You can still override everything via CLI flags:
    python tools/tune_profiles.py \
        --paths data/*.csv \
        --time-col BJD_TDB --flux-col rel_flux \
        --methods oot-replace prewhiten \
        --durations 0.08 0.12 0.20 \
        --n-periods 4000 \
        --null-trials 200 \
        --out profile_overrides.json
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Wire up local src/ for "python tools/tune_profiles.py" from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import transitbench as tb  # noqa: E402
from transitbench.recover import (  # noqa: E402
    PROFILES,
    _compute_baseline_bls,
    basic_features,
    get_clean_series,
)


# ------------------------
# Auto-discovery helpers
# ------------------------
TIME_CANDIDATES = [
    "BJD_TDB", "BTJD", "TIME", "time", "Time", "jd", "JD", "HJD", "BMJD_TDB",
]
FLUX_CANDIDATES = [
    "rel_flux", "flux", "Flux", "FLUX",
    "PDCSAP_FLUX", "SAP_FLUX", "sap_flux", "pdcsap_flux",
    "f", "flux_norm",
]

SEARCH_DIRS = [
    ROOT / "data",
    ROOT / "data" / "raw",
    ROOT / "examples" / "data",
]

EXTS = (".csv", ".fits", ".fit", ".fz", ".tbl")


def find_candidate_paths(max_files: int = 24) -> List[str]:
    paths: List[str] = []
    for d in SEARCH_DIRS + [Path.cwd() / "data", Path.cwd()]:
        if not d.exists():
            continue
        for ext in EXTS:
            paths.extend([str(p) for p in d.rglob(f"*{ext}")])
    # De-dup while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:max_files]


def load_any(path: str, time_hint: Optional[str] = None, flux_hint: Optional[str] = None):
    """
    Try multiple (time, flux) column names against tb.load until one works.
    Returns (lc, time_col_used, flux_col_used).
    """
    # If explicit hints are provided, try them first.
    pairs: List[Tuple[str, str]] = []
    if time_hint and flux_hint:
        pairs.append((time_hint, flux_hint))

    # Compose candidate pairs (front-load the most common)
    for tname in TIME_CANDIDATES:
        for fname in FLUX_CANDIDATES:
            pairs.append((tname, fname))

    err_last: Optional[Exception] = None
    for tcol, fcol in pairs:
        try:
            lc = tb.load(path, time=tcol, flux=fcol)
            return lc, tcol, fcol
        except Exception as e:  # try next pair
            err_last = e
            continue

    # Final attempt: let tb.load try its own defaults (if it supports autodetect)
    try:
        lc = tb.load(path)
        return lc, "<auto>", "<auto>"
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load '{path}' with any of the known time/flux columns. "
        f"Last error: {err_last}"
    )


# ------------------------
# Null-trial machinery
# ------------------------
def sample_null_snr(t: np.ndarray,
                    f: np.ndarray,
                    method: str,
                    durations: Tuple[float, ...],
                    n_periods: int,
                    n_trials: int,
                    rng: Optional[int] = None) -> np.ndarray:
    """
    Return array of top-z (null) from repeated clean+BLS with no injection,
    using the SAME metric as the detection decision ('depth/err' a.k.a. zscore).
    """
    z = []
    for _ in range(int(n_trials)):
        # Clean series (deterministic except resample-oot modes)
        t_c, f_c, _ = get_clean_series(t, f, method=method, n_periods=n_periods)

        # Compute best BLS solution
        best = _compute_baseline_bls(t_c, f_c, durations=durations, n_periods=n_periods)

        # Prefer explicit depth/err; fallback to z_top only if needed
        depth = best.get("depth", None)
        derr  = best.get("depth_err", None)

        if depth is not None and derr is not None and np.isfinite(depth) and np.isfinite(derr) and derr > 0:
            zz = float(depth / derr)
        else:
            zz = float(best.get("z_top", np.nan))

        if np.isfinite(zz):
            z.append(zz)

    return np.array(z, float)


def aggregate_null(paths: List[str],
                   method: str,
                   durations: Tuple[float, ...],
                   n_periods: int,
                   n_trials: int,
                   time_col: Optional[str] = None,
                   flux_col: Optional[str] = None,
                   max_per_curve: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Dict[str, str]]]:
    """
    Build a single concatenated null distribution across many light curves.
    Returns (null_values, used_columns_per_path)
    """
    used_cols: Dict[str, Dict[str, str]] = {}
    allz: List[np.ndarray] = []

    for p in paths:
        try:
            lc, t_used, f_used = load_any(p, time_hint=time_col, flux_hint=flux_col)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue

        used_cols[p] = {"time": t_used, "flux": f_used}
        z = sample_null_snr(lc.t, lc.f, method=method,
                            durations=durations, n_periods=n_periods,
                            n_trials=n_trials)
        allz.append(z)

    if not allz:
        return np.array([], float), used_cols
    return np.concatenate(allz), used_cols


def choose_tau_from_null(null_z: np.ndarray, fpr_target: float) -> Optional[float]:
    """
    Quantile of null distribution that gives desired per-trial FPR.
    """
    if null_z.size == 0:
        return None
    q = np.clip(1.0 - float(fpr_target), 0.0, 1.0)
    return float(np.quantile(null_z, q))


# ------------------------
# CLI
# ------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tune TransitBench profile thresholds from null trials (no injection). "
                    "With no flags, this script auto-discovers data under ./data and writes "
                    "profile_overrides.json in the project root."
    )
    ap.add_argument("--paths", nargs="+", default=None,
                    help="CSV/FITS/TBL light curves to use for null trials. "
                         "If omitted, discover under ./data.")
    ap.add_argument("--time-col", default=None,
                    help="Time column name (optional, auto-detected otherwise).")
    ap.add_argument("--flux-col", default=None,
                    help="Flux column name (optional, auto-detected otherwise).")
    ap.add_argument("--methods", nargs="+", default=["oot-replace", "prewhiten"])
    ap.add_argument("--durations", nargs="+", type=float, default=[0.08, 0.12, 0.20])
    ap.add_argument("--n-periods", type=int, default=4000)
    ap.add_argument("--null-trials", type=int, default=200,
                    help="Number of null trials per light curve.")
    ap.add_argument("--max-files", type=int, default=24,
                    help="Max number of files to auto-include when --paths is omitted.")
    ap.add_argument("--results", default=None,
                    help="Optional results.json with prior injection runs (for reporting).")
    ap.add_argument("--out", default=str(ROOT / "profile_overrides.json"),
                    help="Where to write suggested profile tau_base values.")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    # Discover files if none provided
    if args.paths is None:
        paths = find_candidate_paths(max_files=args.max_files)
        if not paths:
            print("No input files found under ./data. Please pass --paths.")
            sys.exit(2)
        print(f"Discovered {len(paths)} file(s):")
        for p in paths:
            print("  -", p)
    else:
        paths = []
        for pat in args.paths:
            # Support glob patterns even when passed explicitly
            expanded = glob.glob(pat)
            paths.extend(expanded if expanded else [pat])
        print(f"Using {len(paths)} file(s) from --paths")

    durations = tuple(float(d) for d in args.durations)
    n_periods = int(args.n_periods)
    n_trials = int(args.null_trials)

    suggestions = {}
    report = {"by_method": {}, "used": {"files": paths, "durations": durations, "n_periods": n_periods,
                                        "null_trials_per_curve": n_trials}}

    for method in args.methods:
        print(f"\n== Building null distribution for method '{method}' ==")
        null_z, used_cols = aggregate_null(paths, method=method, durations=durations,
                                           n_periods=n_periods, n_trials=n_trials,
                                           time_col=args.time_col, flux_col=args.flux_col)
        report["by_method"][method] = {
            "null_count": int(null_z.size),
            "null_mean": float(np.nanmean(null_z)) if null_z.size else None,
            "null_std": float(np.nanstd(null_z)) if null_z.size else None,
            "used_columns": used_cols,
        }

        # Propose tau per profile to meet fpr_target
        tau_suggest = {}
        for name, prof in PROFILES.items():
            fpr = prof.get("fpr_target", 0.01)  # default 1% per-trial FPR if unspecified
            tau = choose_tau_from_null(null_z, fpr)
            if tau is not None and math.isfinite(tau):
                tau_suggest[name] = tau

        suggestions[method] = tau_suggest
        print("  Suggested tau_base by profile:")
        for k, v in tau_suggest.items():
            print(f"   - {k:>9}: {v:.2f}")

    # Optional: summarize existing injection results (TPR vs method)
    if args.results and Path(args.results).exists():
        try:
            with open(args.results, "r") as f:
                res = json.load(f)
            tpr_by_method = {}
            blocks = res if isinstance(res, list) else [res]
            for blk in blocks:
                methods = blk.get("methods", {})
                for mname, mblock in methods.items():
                    det = mblock.get("detected", 0)
                    n_inj = mblock.get("n_injections", 0)
                    if n_inj > 0:
                        tpr_by_method.setdefault(mname, []).append(det / n_inj)
            report["tpr_summary"] = {m: float(np.mean(v)) for m, v in tpr_by_method.items()}
        except Exception as e:
            report["tpr_summary_error"] = str(e)

    out_payload = {
        "suggest_tau_base": suggestions,
        "report":report
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote suggestions to: {out_path}")
    print("\nYou can merge these into recover.PROFILES['<profile>']['tau_base'] per method, e.g.")
    print("  PROFILES['balanced']['tau_base']['oot-replace'] = <value>")
    print("  PROFILES['balanced']['tau_base']['prewhiten']  = <value>")


if __name__ == "__main__":
    main()