# transitbench/batch.py
from __future__ import annotations

import os
import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .core import load


def _summarize_one_result(res: Dict[str, Any], label: str) -> str:
    """One-target compact text block."""
    lines: list[str] = []
    lines.append(f"[{label}]")
    basic = res.get("basic", {})
    lines.append(
        f"  n_pts={basic.get('n_pts', '—')}, rms_oot={basic.get('rms_oot', '—')}, beta10={basic.get('beta10', '—')}"
    )
    for m, blk in res.get("methods", {}).items():
        metrics = blk.get("metrics", {})
        # fallback to records-derived metrics if missing
        if not metrics:
            recs = blk.get("records", [])
            det = np.array([1 if (isinstance(r, dict) and r.get("detected")) else 0 for r in recs], dtype=int)
            snr = np.array([r.get("snr_injected", np.nan) for r in recs], dtype=float)
            metrics = dict(
                n_injections=len(recs),
                detected=int(det.sum()),
                tpr=float(det.mean()) if len(det) else 0.0,
                snr_mean=float(np.nanmean(snr)) if np.size(snr) else np.nan,
                snr_median=float(np.nanmedian(snr)) if np.size(snr) else np.nan,
            )
        n = metrics.get("n_injections", 0) or 0
        d = metrics.get("detected", 0) or 0
        tpr = (d / n) if n else 0.0
        smean = metrics.get("snr_mean", np.nan)
        smed = metrics.get("snr_median", np.nan)
        lines.append(
            f"  {m:11s}: Detected {d}/{n} ({tpr:.3f}), mean_z={smean:.2f}, median_z={smed:.2f}"
        )
    return "\n".join(lines)


# ------------- worker (must be top-level for multiprocessing) -------------
def _batch_worker(job: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Execute one scoring job. Returns (path, result_dict).
    """
    lc = load(job["path"], time=job["time_col"], flux=job["flux_col"], label=job["label"])

    res = lc.benchmark(
        method=job["method"],
        compare_with=job["compare_modes"],
        durations=job["durations"],
        depths=job["depths"],
        periods=job["periods"],
        decision_metric=job["decision_metric"],
        snr_threshold=job["snr_thresh"],
        rel_window=job["rel_window"],
        per_injection_search=job["per_injection_search"],
        compute_budget=job["compute_budget"],
        cost_model=job["cost_model"],
        oot_strategy=job["oot_strategy"],
        profile=job["profile"],
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
    time_col: str = "time",
    flux_col: str = "flux",
    method: str = "oot-replace",
    compare_modes: Optional[Sequence[str]] = None,
    durations: Sequence[float] = (0.08, 0.12, 0.20),
    depths: Sequence[float] = (0.003, 0.005, 0.01),
    period_min: float = 0.5,
    period_max: float = 20.0,
    n_periods: int = 10000,
    decision_metric: str = "zscore",
    snr_thresh: float = 6.0,
    rel_window: float = 0.015,
    per_injection_search: bool = True,
    oot_strategy: str = "median",
    compute_budget: Optional[int] = None,
    cost_model: Any = None,
    profile: Optional[str] = None,
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

    periods = np.linspace(float(period_min), float(period_max), int(n_periods))

    job_proto = dict(
        time_col=time_col,
        flux_col=flux_col,
        label=None,
        method=method,
        compare_modes=list(compare_modes or []),
        durations=tuple(durations),
        depths=tuple(depths),
        periods=periods,
        decision_metric=decision_metric,
        snr_thresh=snr_thresh,
        rel_window=rel_window,
        per_injection_search=per_injection_search,
        compute_budget=compute_budget,
        cost_model=cost_model,
        oot_strategy=oot_strategy,
        profile=profile,
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
                j["label"] = os.path.basename(p)
                futs.append(ex.submit(_batch_worker, j))
            for fut in as_completed(futs):
                path, res = fut.result()
                results[path] = res
    else:
        for p in files:
            j = dict(job_proto)
            j["path"] = p
            j["label"] = os.path.basename(p)
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
