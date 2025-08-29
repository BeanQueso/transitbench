from __future__ import annotations
import argparse
from .batch import run_batch

def main():
    p = argparse.ArgumentParser(description="Run TransitBench over many light curves (batch mode).")
    p.add_argument("inputs", nargs="+", help="Files or globs, e.g. 'data/**/*.csv'")
    p.add_argument("--out-root", required=True, help="Root folder for grouped results")
    p.add_argument("--time-col", default=None)
    p.add_argument("--flux-col", default=None)
    p.add_argument("--flux-err-col", default=None)
    p.add_argument("--clean-mode", choices=["prewhiten","oot-replace","mask-only"], default="prewhiten")
    p.add_argument("--compare-modes", default="", help="Comma list, e.g. 'oot-replace'")
    p.add_argument("--window-days", type=float, default=0.5)
    p.add_argument("--mask-pad", type=float, default=2.0)
    p.add_argument("--no-secondary", action="store_true")
    p.add_argument("--snr-thresh", type=float, default=6.0)
    p.add_argument("--compute-budget", type=int, default=0)
    p.add_argument("--period-range", default="0.5,20.0")
    p.add_argument("--durations", default="0.08,0.12,0.20")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-cleaned-csv", action="store_true")
    p.add_argument("--workers", type=int, default=0, help="Parallel workers (default: CPU count)")
    p.add_argument("--no-resume", action="store_true", help="Re-run even if outputs exist")
    args = p.parse_args()

    compare = [m.strip() for m in args.compare_modes.split(",") if m.strip()]
    pmin, pmax = (float(x) for x in args.period_range.split(","))
    durs = [float(x) for x in args.durations.split(",") if x.strip()]

    out = run_batch(
        inputs=args.inputs,
        out_root=args.out_root,
        time_col=args.time_col,
        flux_col=args.flux_col,
        flux_err_col=args.flux_err_col,
        clean_mode=args.clean_mode,
        compare_modes=compare,
        window_days=args.window_days,
        mask_pad=args.mask_pad,
        mask_secondary=not args.no_secondary,
        snr_thresh=args.snr_thresh,
        compute_budget=args.compute_budget,
        period_range=(pmin, pmax),
        durations=durs,
        save_plots=not args.no_plots,
        export_cleaned_csv=not args.no_cleaned_csv,
        n_workers=(args.workers if args.workers > 0 else (0)),
        resume=(not args.no_resume),
    )
    print("Batch outputs:")
    for k, v in out.items():
        print(f"- {k}: {v}")
