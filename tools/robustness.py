#/transitbench/tools/robustness.py
from __future__ import annotations
import os, json, math
from typing import Iterable, Dict, Any, List, Optional, Tuple
import numpy as np

import transitbench as tb
from transitbench import recover
from transitbench.utils import ensure_dir, slugify, write_json
from transitbench.recover import inject_and_recover

# --- minimal reader to support AIJ .tbl and CSV (kept local to avoid paper/ dependency) ---
POSS_TIME = (
    "BJD_TDB","HJD_UTC","JD_UTC","J.D.-2400000","BTJD","BJD","TIME","T_TIME","t_bjd","bjd","btjd","time"
)
POSS_FLUX = (
    "rel_flux_T1",
    "flux","pdcsap_flux","sap_flux","pdcsap_fluxdtr","FLUX","SAP_FLUX","PDCSAP_FLUX"
)

def _read_table_arrays(path: str):
    import numpy as _np, os as _os
    from pathlib import Path as _Path
    p = _Path(path)
    is_tbl = p.suffix.lower() == ".tbl"
    try:
        if is_tbl:
            dat = _np.genfromtxt(path, names=True, dtype=None, encoding=None)
        else:
            dat = _np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        if is_tbl:
            dat = _np.genfromtxt(path, names=True)
        else:
            dat = _np.genfromtxt(path, delimiter=",")
    names = tuple(dat.dtype.names or ())
    lower = {n.lower(): n for n in names}
    def pick(cands):
        for c in cands:
            k = c.lower()
            if k in lower:
                return lower[k]
        return None
    tkey = pick(POSS_TIME); fkey = pick(POSS_FLUX)
    if not tkey or not fkey:
        raise ValueError(f"Could not find time/flux columns in {_os.path.basename(path)}. Detected columns: {list(names)}")
    t = _np.asarray(dat[tkey], float)
    f = _np.asarray(dat[fkey], float)
    if tkey.lower() == "j.d.-2400000":
        t = t + 2400000.0
    m = _np.isfinite(t) & _np.isfinite(f)
    return t[m], f[m]

def robustness_curve_for_file(
    path: str,
    *,
    methods: Iterable[str] = ("oot-replace", "prewhiten"),
    profile: str = "balanced",
    epsilons: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    n_injections: int = 27,
    durations=(0.08, 0.12, 0.20),
    depths=(0.003, 0.005, 0.010),
    periods=(1.5, 3.0, 5.0),
    per_injection_search: bool = True,
    decision_metric: str = "zscore",
    seed: int = 123,
    compute_budget: Optional[int] = None,
    cost_model: Optional["recover.CostModel"] = None,
    label_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For one LC file, sweep adversarial eps and measure TPR per method.
    Returns a dict:
      {
        "label": ..., "profile": "balanced", "epsilons":[...],
        "methods": {
           "oot-replace": {"tpr":[...], "snr_mean":[...], "snr_median":[...]},
           "prewhiten":   {...}
        }
      }
    """
    import numpy as _np
    import os as _os

    # Load time/flux arrays (supports CSV and AIJ .tbl)
    t, f = _read_table_arrays(path)
    label = label_override or _os.path.splitext(_os.path.basename(path))[0]

    results: Dict[str, Any] = {"label": label, "profile": profile, "epsilons": list(epsilons), "methods": {}}

    rng = _np.random.RandomState(seed)

    for m in methods:
        tprs, snr_means, snr_meds = [], [], []
        for eps in epsilons:
            try:
                block = inject_and_recover(
                    t, f,
                    method=m,
                    profile=profile,
                    depths=list(depths),
                    durations=list(durations),
                    periods=list(periods),
                    compute_budget=compute_budget,
                    n_periods=4000,
                    adversarial=True,
                    adversarial_eps=float(eps),
                    adversarial_seed=int(rng.randint(0, 1_000_000)),
                    cache_clean_root=_os.path.expanduser("~/.cache/transitbench/clean"),
                    show_progress=False,
                    label=label,
                )
                recs = block.get("records", [])
                if recs:
                    det = sum(1 for r in recs if r.get("detected"))
                    tpr = det / max(1, len(recs))
                    snr_vals = [float(r.get("snr_injected", _np.nan)) for r in recs if "snr_injected" in r]
                    snr_vals = [x for x in snr_vals if _np.isfinite(x)]
                    snr_mean = float(_np.mean(snr_vals)) if snr_vals else float("nan")
                    snr_median = float(_np.median(snr_vals)) if snr_vals else float("nan")
                else:
                    tpr, snr_mean, snr_median = 0.0, float("nan"), float("nan")
            except Exception:
                tpr, snr_mean, snr_median = 0.0, float("nan"), float("nan")
            tprs.append(tpr); snr_means.append(snr_mean); snr_meds.append(snr_median)

        results["methods"][m] = {"tpr": tprs, "snr_mean": snr_means, "snr_median": snr_meds}

    out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "runs", "robustness"))
    out_path = os.path.abspath(os.path.join(out_dir, f"{slugify(label)}__{profile}.json"))
    write_json(results, out_path, indent=2)
    return results

def run_robustness_for_list(
    paths: Iterable[str],
    *,
    methods: Iterable[str] = ("oot-replace", "prewhiten"),
    profile: str = "balanced",
    epsilons: Tuple[float, ...] = (0.0, 0.5, 1.0),
    durations: Tuple[float, ...] = (0.08, 0.12, 0.20),
    depths: Tuple[float, ...] = (0.003, 0.005, 0.010),
    periods: Tuple[float, ...] = (1.5, 3.0, 5.0),
    compute_budget: Optional[int] = 600,
    n_injections: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> List[str]:
    """
    Tiny adversarial sweep for each path; writes one JSON per method per file.
    Returns list of written filepaths.
    """
    out_dir = ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "robustness"))
    written: List[str] = []

    grid_d = list(durations)
    grid_dep = list(depths)
    grid_p = list(periods)

    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            t, f = _read_table_arrays(path)
        except Exception as e:
            print(f"[robustness] {os.path.basename(path)} -> read error: {e}")
            continue

        for eps in epsilons:
            for m in methods:
                try:
                    block = inject_and_recover(
                        t, f,
                        method=m,
                        profile=profile,
                        depths=grid_dep,
                        durations=grid_d,
                        periods=grid_p,
                        compute_budget=compute_budget,
                        n_periods=4000,
                        adversarial=True,
                        adversarial_eps=float(eps),
                        adversarial_seed=42,
                        cache_clean_root=os.path.expanduser("~/.cache/transitbench/clean"),
                        show_progress=False,
                        label=base,
                    )
                    payload = {
                        "label": base,
                        "profile": profile,
                        "epsilon": float(eps),
                        "method": m,
                        "result": block,
                    }
                    out_path = os.path.join(out_dir, f"{slugify(base)}__{m}__eps{eps:.2f}.json")
                    write_json(payload, out_path, indent=2)
                    written.append(out_path)
                except Exception as e:
                    print(f"[robustness] {os.path.basename(path)} -> error: {e}")

    return written