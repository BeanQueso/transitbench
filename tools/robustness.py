# /Users/eshaantripathi/Documents/transitbench/tools/robustness.py
from __future__ import annotations
import os, json, math
from typing import Iterable, Dict, Any, List, Optional, Tuple
import numpy as np

import transitbench as tb
from transitbench import recover
from transitbench.utils import ensure_dir, slugify, write_json

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
           "oot-replace": {"tpr":[...], "snr_mean":[...], "snr_med":[...]},
           "prewhiten":   {...}
        }
      }
    """
    # Load LC using the same loader (it already handles CSV/FITS/TBL in your codebase)
    lc = tb.load(path)
    label = label_override or (lc.label or os.path.splitext(os.path.basename(path))[0])

    results: Dict[str, Any] = {"label": label, "profile": profile, "epsilons": list(epsilons), "methods": {}}

    # shared knobs for fairness (same grid / same budget)
    shared = dict(
        per_injection_search=per_injection_search,
        decision_metric=decision_metric,
        durations=list(durations),
        depths=list(depths),
        periods=list(periods),
        adversarial=True,
        compute_budget=compute_budget,
        cost_model=cost_model,
    )

    rng = np.random.RandomState(seed)

    for m in methods:
        tprs, snr_means, snr_meds = [], [], []
        for eps in epsilons:
            res = lc.benchmark(
                method=(m,),
                profile=profile,
                adversarial_eps=float(eps),
                adversarial_seed=int(rng.randint(0, 1_000_000)),
                **shared
            )
            block = res["methods"][m]
            recs = block.get("records", [])
            if recs:
                detected = sum(1 for r in recs if r.get("detected"))
                tpr = detected / max(1, len(recs))
                snr_vals = [float(r.get("snr_injected", np.nan)) for r in recs if "snr_injected" in r]
                snr_vals = [x for x in snr_vals if not (x is None or math.isnan(x))]
                snr_mean = float(np.mean(snr_vals)) if snr_vals else float("nan")
                snr_median = float(np.median(snr_vals)) if snr_vals else float("nan")
            else:
                tpr, snr_mean, snr_median = 0.0, float("nan"), float("nan")

            tprs.append(tpr); snr_means.append(snr_mean); snr_meds.append(snr_median)

        results["methods"][m] = {
            "tpr": tprs,
            "snr_mean": snr_means,
            "snr_median": snr_meds,
        }

    # persist for reproducibility
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
    n_injections: Optional[int] = None,   # inferred from grid if None
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
        try:
            lc = tb.load(path)
        except Exception as e:
            print(f"[robustness] {os.path.basename(path)} -> load error: {e}")
            continue

        for eps in epsilons:
            shared = dict(
                per_injection_search=True,
                decision_metric="zscore",
                durations=grid_d,
                depths=grid_dep,
                periods=grid_p,
                compute_budget=compute_budget,
                adversarial=True,
                adversarial_eps=eps,
                adversarial_seed=42,
            )

            for m in methods:
                try:
                    res = lc.benchmark(method=(m,), profile=profile, **shared)
                    payload = {
                        "label": lc.label,
                        "profile": profile,
                        "epsilon": eps,
                        "method": m,
                        "result": res["methods"][m],
                    }
                    out_path = os.path.join(
                        out_dir,
                        f"{slugify(lc.label)}__{m}__eps{eps:.2f}.json"
                    )
                    write_json(payload, out_path, indent=2)
                    written.append(out_path)
                except Exception as e:
                    print(f"[robustness] {os.path.basename(path)} -> error: {e}")

    return written