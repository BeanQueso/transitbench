# transitbench/batch.py
from __future__ import annotations

import os
import sys
import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .io import load_timeseries
from .core import LightCurve
from .api import to_text_report


def _summarize_one_result(res: Dict[str, Any], label: str) -> str:
    """One-target compact text block."""
    lines: list[str] = []
    lines.append(f"[{label}]")
    basic = res.get("basic", {})
    lines.append(
        f"  n_pts={basic.get('n_pts', '—')}, rms_oot={basic.get('rms_oot', '—')}, beta10={basic.get('beta10', '—')}"
    )
    for m, blk in res.get("methods", {}).items():
        n = blk.get("n_injections", 0) or 0
        d = blk.get("detected", 0) or 0
        tpr = (d / n) if n else 0.0
        smean = blk.get("snr_mean", np.nan)
        smed = blk.get("snr_median", np.nan)
        lines.append(
            f"  {m:11s}: Detected {d}/{n} ({tpr:.3f}), mean_z={smean:.2f}, median_z={smed:.2f}"
        )
    return "\n".join(lines)


# ------------- worker (must be top-level for multiprocessing) -------------
def _batch_worker(job: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Execute one scoring job. Returns (path, result_dict).
    """
    from . import recover  # local import inside worker
    # Load LC
    ts = load_timeseries(job["path"], time_col=job["time_col"], flux_col=job["flux_col"])
    lc = LightCurve(time=ts["time"], flux=ts["flux"], meta={"source_path": job["path"]})

    # Run benchmark via recover to avoid circulars
    res = recover.inject_and_recover(
        np.asarray(lc.time, float),
        np.asarray(lc.flux, float),
        method=job["method"],
        compare_with=job["compare_modes"],
        durations=job["durations"],
        depths=job["depths"],
        inj_periods=job["inj_periods"],
        inj_durations=job["inj_durations"],
        periods_grid=job["periods_grid"],
        window_days=job["window_days"],
        mask_pad=job["mask_pad"],
        baseline_period=job["baseline_period"],
        baseline_t0=job["baseline_t0"],
        baseline_duration=job["baseline_duration"],
        decision_metric=job["decision_metric"],
        snr_thresh=job["snr_thresh"],
        rel_window=job["rel_window"],
        allow_harmonics=True,
        per_injection_search=job["per_injection_search"],
        compute_budget=job["compute_budget"],
        cost_model=job["cost_model"],
        oot_strategy=job["oot_strategy"],
        profile=job["profile"],
        profile_overrides=job["profile_overrides"],
    )
    return job["path"], res


def _expand_inputs(inputs: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pat in inputs:
        if any(ch in pat for ch in "*?[]"):
            paths.extend(glob.glob(pat, recursive=True))
        else:
            paths.append(pat)
    # De-dup & keep stable order
    seen = set()
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def run_batch(
    *,
    inputs: Sequence[str],
    out_root: str,
    time_col: str,
    flux_col: str,
    method: str = "oot-replace",
    compare_modes: Optional[Sequence[str]] = None,
    durations: Sequence[float] = (0.08, 0.12, 0.20),
    depths: Sequence[float] = (0.003, 0.005, 0.01),
    inj_periods: Sequence[float] = (1.5, 3.0, 5.0),
    inj_durations: Optional[Sequence[float]] = None,
    periods_grid: Optional[np.ndarray] = None,
    decision_metric: str = "zscore",
    snr_thresh: float = 6.0,
    rel_window: float = 0.015,
    per_injection_search: bool = True,
    window_days: float = 0.3,
    mask_pad: float = 3.0,
    oot_strategy: str = "median",
    baseline_period: Optional[float] = None,
    baseline_t0: Optional[float] = None,
    baseline_duration: Optional[float] = None,
    compute_budget: Optional[int] = None,
    cost_model: Any = None,
    profile: Optional[str] = None,
    profile_overrides: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    save_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a batch of files and optionally save a single text report.

    Returns a dict:
      { "files": [...], "results": {path: result_dict, ...}, "summary_text": "..."}
    """
    files = _expand_inputs(list(inputs))
    os.makedirs(out_root, exist_ok=True)
    if save_to is None:
        save_to = os.path.join(out_root, "batch_report.txt")

    print(f"[TransitBench] Matched {len(files)} files.")
    modes = [method] + list(compare_modes or [])
    print(f"[TransitBench] Modes: {modes}")
    print(f"[TransitBench] Output root: {out_root}")
    print(f"[TransitBench] Using {n_workers} worker(s).")

    job_proto = dict(
        time_col=time_col,
        flux_col=flux_col,
        method=method,
        compare_modes=list(compare_modes or []),
        durations=tuple(durations),
        depths=tuple(depths),
        inj_periods=tuple(inj_periods),
        inj_durations=None if inj_durations is None else tuple(inj_durations),
        periods_grid=periods_grid,
        window_days=window_days,
        mask_pad=mask_pad,
        baseline_period=baseline_period,
        baseline_t0=baseline_t0,
        baseline_duration=baseline_duration,
        decision_metric=decision_metric,
        snr_thresh=snr_thresh,
        rel_window=rel_window,
        per_injection_search=per_injection_search,
        compute_budget=compute_budget,
        cost_model=cost_model,
        oot_strategy=oot_strategy,
        profile=profile,
        profile_overrides=profile_overrides or {},
    )

    results: Dict[str, Any] = {}
    if len(files) == 0:
        summary = "No files matched."
        if save_to:
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(summary + "\n")
        return {"files": [], "results": {}, "summary_text": summary}

    # Multiprocessing or serial
    if n_workers and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for p in files:
                j = dict(job_proto)
                j["path"] = p
                futs.append(ex.submit(_batch_worker, j))
            for fut in as_completed(futs):
                path, res = fut.result()
                results[path] = res
    else:
        for p in files:
            j = dict(job_proto)
            j["path"] = p
            path, res = _batch_worker(j)
            results[path] = res

    # Build a single text summary
    lines: list[str] = []
    header = "TransitBench — batch injection–recovery"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(f"Files: {len(files)}")
    lines.append(f"Profile: {profile or '—'} ; Method(s): {', '.join(modes)}")
    lines.append("")

    # Per-file summaries
    for p in files:
        label = os.path.basename(p)
        if p in results:
            lines.append(_summarize_one_result(results[p], label))
            lines.append("")
        else:
            lines.append(f"[{label}]\n  ERROR: no result produced.\n")

    summary_text = "\n".join(lines).rstrip() + "\n"
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[TransitBench] Wrote summary to: {save_to}")

    return {"files": files, "results": results, "summary_text": summary_text}
