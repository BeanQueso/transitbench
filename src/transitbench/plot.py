from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def cleanmode_suffix(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode == "oot-replace":
        return "oot-replaced"
    if mode == "prewhiten":
        return "prewhitened"
    if mode == "mask-only":
        return "mask-only"
    return mode or "cleaned"

def plot_timeseries(time, flux, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(np.asarray(time), np.asarray(flux), ".", ms=2, alpha=0.8, label=label)
    if label:
        ax.legend(loc="best")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative flux")
    return fig, ax

def save_raw_and_clean(
    time_raw, flux_raw,
    time_clean, flux_clean,
    out_dir: str,
    base: str,
    clean_mode: str,
    save_raw: bool = True,
):
    """
    Save raw and cleaned light curve PNGs with consistent names into out_dir.
    - raw plot:     <base>_raw.png
    - cleaned plot: <base>_<suffix>.png  where suffix = cleanmode_suffix(clean_mode)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    suffix = cleanmode_suffix(clean_mode)

    # RAW
    raw_path = None
    if save_raw:
        fig, ax = plt.subplots()
        try:
            plot_timeseries(time_raw, flux_raw, ax=ax)
            fig.tight_layout()
            raw_path = os.path.join(out_dir, f"{base}_raw.png")
            fig.savefig(raw_path, dpi=160)
        finally:
            plt.close(fig)

    # CLEANED
    fig, ax = plt.subplots()
    try:
        plot_timeseries(time_clean, flux_clean, ax=ax)
        fig.tight_layout()
        cln_path = os.path.join(out_dir, f"{base}_{suffix}.png")
        fig.savefig(cln_path, dpi=160)
    finally:
        plt.close(fig)

    return {"raw_png": raw_path, "cleaned_png": os.path.join(out_dir, f"{base}_{suffix}.png")}
