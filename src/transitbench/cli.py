# transitbench/cli.py
from __future__ import annotations

import sys
import os
import argparse
import numpy as np
from typing import Optional, Sequence

from . import api
from .core import load
from .batch import run_batch


# ----------------------
# Single-file scoring
# ----------------------
def main_score(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tb-score",
        description="Score a single light curve via injection–recovery (text-only).",
    )
    ap.add_argument("path", help="Path to light curve table (CSV).")
    ap.add_argument("--time", default="time", help="Time column name (default: time).")
    ap.add_argument("--flux", default="flux", help="Flux column name (default: flux).")
    ap.add_argument("--method", default="oot-replace", choices=["oot-replace", "prewhiten"])
    ap.add_argument("--compare-with", default="", help="Comma-separated list of other methods to compare (e.g., prewhiten)")
    ap.add_argument("--durations", default="0.08,0.12,0.20", help="Durations grid (days), comma-separated")
    ap.add_argument("--depths", default="0.003,0.005,0.01", help="Injection depths, comma-separated")
    ap.add_argument("--inj-periods", default="1.5,3.0,5.0", help="Injection periods (days), comma-separated")
    ap.add_argument("--inj-durations", default="", help="Injection durations (days), comma-separated (default: same as --durations)")
    ap.add_argument("--period-min", type=float, default=0.5)
    ap.add_argument("--period-max", type=float, default=20.0)
    ap.add_argument("--n-periods", type=int, default=10000)

    ap.add_argument("--decision-metric", default="zscore", choices=["zscore", "depth_over_err"])
    ap.add_argument("--snr-thresh", type=float, default=6.0)
    ap.add_argument("--rel-window", type=float, default=0.015)

    ap.add_argument("--per-injection-search", action="store_true", default=True)
    ap.add_argument("--no-per-injection-search", dest="per_injection_search", action="store_false")

    ap.add_argument("--oot-strategy", default="median", choices=["median", "resample"])
    ap.add_argument("--profile", choices=["sensitive", "balanced", "strict"], default=None)

    ap.add_argument("--save-report", default="", help="Optional path to save a text report.")
    args = ap.parse_args(argv)

    durations = tuple(float(x) for x in args.durations.split(",") if x.strip())
    depths = tuple(float(x) for x in args.depths.split(",") if x.strip())
    inj_periods = tuple(float(x) for x in args.inj_periods.split(",") if x.strip())
    if args.inj_durations.strip():
        # kept for compatibility; currently not used directly
        _ = tuple(float(x) for x in args.inj_durations.split(",") if x.strip())

    compare_with = [s.strip() for s in args.compare_with.split(",") if s.strip()]

    lc = load(args.path, time=args.time, flux=args.flux)

    res = lc.benchmark(
        method=args.method,
        compare_with=compare_with,
        durations=durations,
        depths=depths,
        periods=inj_periods,
        decision_metric=args.decision_metric,
        snr_threshold=args.snr_thresh,
        rel_window=args.rel_window,
        per_injection_search=args.per_injection_search,
        oot_strategy=args.oot_strategy,
        profile=args.profile,
        compute_budget=None,
        cost_model=None,
    )

    header = "TransitBench — injection–recovery (CLI)"
    api.print_report(res, header=header)

    if args.save_report:
        api.save_report(res, args.save_report, header=header)
        print(f"[TransitBench] Saved: {os.path.abspath(args.save_report)}")

    return 0


# ----------------------
# Batch scoring
# ----------------------
def main_batch(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tb-batch",
        description="Run a batch of light curves and save a single text report.",
    )
    ap.add_argument("inputs", nargs="+", help="One or more paths/globs (e.g., '/data/**/*.csv')")
    ap.add_argument("--out-root", required=True, help="Directory to write the batch_report.txt")
    ap.add_argument("--time", default="time", help="Time column name")
    ap.add_argument("--flux", default="flux", help="Flux column name")

    ap.add_argument("--method", default="oot-replace", choices=["oot-replace", "prewhiten"])
    ap.add_argument("--compare-with", default="", help="Comma-separated other methods to compare")

    ap.add_argument("--durations", default="0.08,0.12,0.20")
    ap.add_argument("--depths", default="0.003,0.005,0.01")
    ap.add_argument("--inj-periods", default="1.5,3.0,5.0")
    ap.add_argument("--inj-durations", default="")

    ap.add_argument("--period-min", type=float, default=0.5)
    ap.add_argument("--period-max", type=float, default=20.0)
    ap.add_argument("--n-periods", type=int, default=10000)

    ap.add_argument("--decision-metric", default="zscore", choices=["zscore", "depth_over_err"])
    ap.add_argument("--snr-thresh", type=float, default=6.0)
    ap.add_argument("--rel-window", type=float, default=0.015)
    ap.add_argument("--window-days", type=float, default=0.30)
    ap.add_argument("--mask-pad", type=float, default=3.0)
    ap.add_argument("--oot-strategy", default="median", choices=["median", "resample"])

    ap.add_argument("--baseline-period", type=float, default=None)
    ap.add_argument("--baseline-t0", type=float, default=None)
    ap.add_argument("--baseline-duration", type=float, default=None)

    ap.add_argument("--profile", choices=["sensitive", "balanced", "strict"], default=None)

    ap.add_argument("--n-workers", type=int, default=1, help="Process workers (1 = serial)")
    ap.add_argument("--save-to", default="", help="Custom path for the single text report. Default: <out_root>/batch_report.txt")

    args = ap.parse_args(argv)

    durations = tuple(float(x) for x in args.durations.split(",") if x.strip())
    depths = tuple(float(x) for x in args.depths.split(",") if x.strip())
    compare_with = [s.strip() for s in args.compare_with.split(",") if s.strip()]
    save_to = args.save_to if args.save_to else None

    _ = run_batch(
        inputs=args.inputs,
        out_root=args.out_root,
        time_col=args.time,
        flux_col=args.flux,
        method=args.method,
        compare_modes=compare_with,
        durations=durations,
        depths=depths,
        period_min=args.period_min,
        period_max=args.period_max,
        n_periods=args.n_periods,
        decision_metric=args.decision_metric,
        snr_thresh=args.snr_thresh,
        rel_window=args.rel_window,
        per_injection_search=args.per_injection_search,
        oot_strategy=args.oot_strategy,
        profile=args.profile,
        n_workers=int(args.n_workers),
        save_to=save_to,
    )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point compatible with both console scripts:
      - tb-score  -> calls main_score
      - tb-batch  -> calls main_batch
    """
    prog = os.path.basename(sys.argv[0]).lower()
    if "tb-batch" in prog:
        return main_batch(argv)
    return main_score(argv)


if __name__ == "__main__":
    raise SystemExit(main())
