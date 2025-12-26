from __future__ import annotations
import argparse
from .batch import run_batch

def main():
    p = argparse.ArgumentParser(description="Run TransitBench over many light curves (batch mode).")
    p.add_argument("inputs", nargs="+", help="Files or globs, e.g. 'data/**/*.csv'")
    p.add_argument("--out-root", required=True, help="Root folder for grouped results")
    p.add_argument("--time-col", default="time")
    p.add_argument("--flux-col", default="flux")
    p.add_argument("--method", default="oot-replace", choices=["oot-replace", "prewhiten"])
    p.add_argument("--compare-modes", default="", help="Comma list, e.g. 'prewhiten'")
    p.add_argument("--durations", default="0.08,0.12,0.20")
    p.add_argument("--depths", default="0.003,0.005,0.01")
    p.add_argument("--period-min", type=float, default=0.5)
    p.add_argument("--period-max", type=float, default=20.0)
    p.add_argument("--n-periods", type=int, default=10000)
    p.add_argument("--decision-metric", default="zscore", choices=["zscore", "depth_over_err"])
    p.add_argument("--snr-thresh", type=float, default=6.0)
    p.add_argument("--rel-window", type=float, default=0.015)
    p.add_argument("--per-injection-search", action="store_true", default=True)
    p.add_argument("--no-per-injection-search", dest="per_injection_search", action="store_false")
    p.add_argument("--oot-strategy", default="median", choices=["median", "resample"])
    p.add_argument("--profile", choices=["sensitive", "balanced", "strict"], default=None)
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    p.add_argument("--save-to", default="", help="Override summary path (default: <out_root>/batch_report.txt)")
    args = p.parse_args()

    compare = [m.strip() for m in args.compare_modes.split(",") if m.strip()]
    durs = [float(x) for x in args.durations.split(",") if x.strip()]
    deps = [float(x) for x in args.depths.split(",") if x.strip()]

    out = run_batch(
        inputs=args.inputs,
        out_root=args.out_root,
        time_col=args.time_col,
        flux_col=args.flux_col,
        method=args.method,
        compare_modes=compare,
        durations=durs,
        depths=deps,
        period_min=args.period_min,
        period_max=args.period_max,
        n_periods=args.n_periods,
        decision_metric=args.decision_metric,
        snr_thresh=args.snr_thresh,
        rel_window=args.rel_window,
        per_injection_search=args.per_injection_search,
        oot_strategy=args.oot_strategy,
        profile=args.profile,
        n_workers=(args.workers if args.workers and args.workers > 0 else 1),
        save_to=(args.save_to or None),
    )
    print("Batch outputs:")
    for k, v in out.items():
        print(f"- {k}: {v}")
