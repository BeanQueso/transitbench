# src/transitbench/io.py
import os
import io
from typing import Optional, Tuple, Dict, Any, List, cast

import numpy as np
import pandas as pd

# FITS is optional; only imported if needed
def _maybe_import_fits():
    try:
        from astropy.io import fits  # type: ignore
        return fits
    except Exception:
        return None


# ------------ utilities ------------

def _find_col(candidates: List[str], columns: List[str]) -> Optional[str]:
    """
    Return the first matching column name (case-insensitive) from candidates.
    """
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _read_text_table(path: str, is_tbl: bool = False) -> "pd.DataFrame":
    """
    Read CSV/TXT/TSV/TBL into a DataFrame, being tolerant to:
      - comment lines (#, !, ;)
      - pipe-delimited .tbl (|…|) or whitespace-delimited .tbl
    Requires pandas. If pandas is unavailable, raises RuntimeError.
    """
    if pd is None:
        raise RuntimeError("pandas is required to read text tables.")

    # Read raw and normalize .tbl with pipes -> whitespace
    if is_tbl:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = []
            for line in fh:
                if not line.strip() or line.lstrip().startswith(("#", "!", ";")):
                    continue
                # strip leading/trailing pipes and collapse pipes to spaces
                s = line.strip().strip("|").replace("|", " ")
                lines.append(s + "\n")
        if not lines:
            raise ValueError(f"No data rows found in {os.path.basename(path)}")
        buf = io.StringIO("".join(lines))
        # Use flexible whitespace separator; header assumed on first non-comment line
        df = pd.read_csv(buf, sep=r"\s+", engine="python")
    else:
        # Let pandas sniff separator; ignore typical comment tokens
        df = pd.read_csv(
            path,
            comment="#",
            sep=None,            # automatic detection
            engine="python",
        )
        # If it looks TSV, try again with \t (pandas' sniffer can miss this)
        if df.shape[1] == 1:
            try:
                df = pd.read_csv(path, comment="#", sep="\t")
            except Exception:
                pass

    # Drop empty unnamed columns created by ragged separators
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    return df


# ------------ public loader ------------

def load_timeseries(
    path: str,
    *,
    # preferred names
    time: Optional[str] = None,
    flux: Optional[str] = None,
    quality: Optional[str] = None,
    label: Optional[str] = None,
    # legacy/synonym names (aliased below)
    time_col: Optional[str] = None,
    flux_col: Optional[str] = None,
    quality_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a light curve time series from CSV/TXT/TSV/TBL or FITS.

    Parameters
    ----------
    path : str
        File path ending with .csv/.txt/.tsv/.tbl or .fits/.fit/.fz
    time, flux : Optional[str]
        Explicit column names to use (case-insensitive). If not provided,
        we try a robust guess.
    quality_col : Optional[str]
        Name of a quality/flags column to filter (kept in meta only; not filtered here).
    label : Optional[str]
        A label to attach in meta; defaults to basename without extension.

    Returns
    -------
    t : np.ndarray
    f : np.ndarray
    meta : dict
        Includes keys: format, source_path, time_col, flux_col, (optionally) quality_col.
    """
    if time is None and time_col is not None:
        time = time_col
    if flux is None and flux_col is not None:
        flux = flux_col
    if quality is None and quality_col is not None:
        quality = quality_col
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    ext = ext.lower()

    meta: Dict[str, Any] = {
        "source_path": os.path.abspath(path),
        "format": ext.lstrip("."),
        "label": label or name,
    }

    # ---------- TEXT TABLES ----------
    if ext in {".csv", ".txt", ".tsv", ".tbl"}:
        is_tbl = (ext == ".tbl")
        df = _read_text_table(path, is_tbl=is_tbl)

        # Column detection
        t_col = time or _find_col(
            [
                "BJD_TDB", "TMID_BJD", "TIME", "BTJD", "HJD", "JD",
                "bjd_tdb", "tmid_bjd", "time", "t",
            ],
            list(df.columns.astype(str))
        )
        f_col = flux or _find_col(
            [
                "rel_flux", "FLUX", "flux", "SAP_FLUX", "PDCSAP_FLUX",
                "NORM_FLUX", "NORMFLUX", "raw_flux", "f",
            ],
            list(df.columns.astype(str))
        )
        if t_col is None or f_col is None:
            raise ValueError(
                f"Could not find time/flux columns in {base}. "
                f"Detected columns: {list(df.columns)}"
            )

        # Cast to numeric & clean
        t = pd.to_numeric(df[t_col], errors="coerce").to_numpy()
        f = pd.to_numeric(df[f_col], errors="coerce").to_numpy()

        finite = np.isfinite(t) & np.isfinite(f)
        t, f = t[finite], f[finite]

        # Attach meta
        meta.update({"time_col": t_col, "flux_col": f_col})
        if quality_col and quality_col in df.columns:
            meta["quality_col"] = quality_col

        return t.astype(float, copy=False), f.astype(float, copy=False), meta

    # ---------- FITS (optionally BTJD) ----------
    if ext in {".fits", ".fit", ".fz"}:
        fits_mod = _maybe_import_fits()
        if fits_mod is None:
            raise RuntimeError("To read FITS files, install astropy (pip install astropy).")

        # cast to Any to keep type checkers happy about context manager methods
        with cast(Any, fits_mod).open(path) as hdul:
            # Try common HDU names for light curves
            # (TESS: 'LIGHTCURVE' or first bin table after primary)
            hdu = None
            for cand in ["LIGHTCURVE", "LC", 1]:
                try:
                    h = hdul[cand]
                    if getattr(h, "data", None) is not None:
                        hdu = h
                        break
                except Exception:
                    continue
            if hdu is None or getattr(hdu, "data", None) is None:
                raise ValueError(f"No light-curve table HDU found in {base}.")

            data = hdu.data
            cols = [c.upper() for c in data.columns.names] if hasattr(data, "columns") else []

            # Select time column
            # Prefer explicit arg; else choose from common names
            def _col_or_none(name: str):
                try:
                    return np.array(data[name])
                except Exception:
                    try:
                        return np.array(data[name.upper()])
                    except Exception:
                        try:
                            return np.array(data[name.lower()])
                        except Exception:
                            return None

            time_array = None
            for cname in ([time] if time else []) + ["BJD_TDB", "TMID_BJD", "TIME", "BTJD", "HJD", "JD"]:
                if cname is None:
                    continue
                arr = _col_or_none(cname)
                if arr is not None:
                    time_array = arr
                    meta["time_col"] = cname
                    break
            if time_array is None:
                raise ValueError(f"Could not find a TIME column in FITS ({cols}).")

            # Flux column
            flux_array = None
            for cname in ([flux] if flux else []) + ["REL_FLUX", "FLUX", "SAP_FLUX", "PDCSAP_FLUX"]:
                if cname is None:
                    continue
                arr = _col_or_none(cname)
                if arr is not None:
                    flux_array = arr
                    meta["flux_col"] = cname
                    break
            if flux_array is None:
                raise ValueError(f"Could not find a FLUX column in FITS ({cols}).")

            # Convert to float & clean
            t = np.array(time_array, dtype=float)
            f = np.array(flux_array, dtype=float)
            finite = np.isfinite(t) & np.isfinite(f)
            t, f = t[finite], f[finite]

            # Note: we do NOT auto-convert BTJD->BJD, but we record it
            if meta.get("time_col", "").upper() == "BTJD":
                meta["time_system"] = "BTJD"

            return t, f, meta

    # Unsupported
    raise ValueError(f"Unsupported file extension '{ext}' for {base}.")
