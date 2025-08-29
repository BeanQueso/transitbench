import os, glob, json
from typing import Optional, List, Any, Dict, Tuple
import numpy as np
import pandas as pd
from transitbench.utils import ensure_dir


def aggregate_runs(runs_root: str, out_csv: Optional[str] = None):
    """
    Collects all injection/recovery records from completeness_*.json under runs_root
    and returns a pandas DataFrame. Optionally writes CSV if out_csv is provided.
    """
    records: List[dict] = []
    for p in glob.glob(os.path.join(runs_root, "completeness_*.json")):
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        results = data.get("results", {})
        for method, d in results.items():
            for r in d.get("records", []):
                row = dict(r)
                row["method"] = row.get("method") or method
                row["depth"] = float(row.get("depth", row.get("inj_depth", np.nan)))
                row["duration"] = float(row.get("duration", row.get("inj_duration", np.nan)))
                det = row.get("detected")
                if isinstance(det, str):
                    det_val = det.strip().lower() in ("1", "true", "yes")
                else:
                    det_val = bool(det)
                row["detected"] = int(det_val)
                records.append(row)

    if not records:
        print(f"[aggregate] no completeness_*.json records found under {runs_root}")
        return None

    df = pd.DataFrame.from_records(records)
    if out_csv:
        ensure_dir(os.path.dirname(out_csv))
        df.to_csv(out_csv, index=False)
        print(f"[aggregate] wrote {out_csv} with {len(df)} rows")
    return df

def _normalize_curves_data(curves: Any) -> Dict[str, List[Tuple[float, float]]]:
    """
    Accept a variety of shapes and return {method: [(eps, tpr), ...]}.
    Supported inputs:
      - { "methodA": [{"eps":0.0,"tpr":1.0}, {"eps":0.5,"tpr":0.9}], "methodB": ... }
      - { "methodA": [(0.0,1.0), (0.5,0.9)], ... }
      - [ {"method":"methodA","eps":0.0,"tpr":1.0}, {"method":"methodA","eps":0.5,"tpr":0.9}, ... ]
      - [ (0.0,1.0), (0.5,0.9) ]  # interpreted as a single unnamed method
    """
    out: Dict[str, List[Tuple[float, float]]] = {}
    if curves is None:
        return out

    # Dict-like input
    if isinstance(curves, dict):
        for m, seq in curves.items():
            pts: List[Tuple[float, float]] = []
            if isinstance(seq, dict) and "eps" in seq and "tpr" in seq:
                pts.append((float(seq["eps"]), float(seq["tpr"])))
            elif isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict) and "eps" in item and "tpr" in item:
                        pts.append((float(item["eps"]), float(item["tpr"])))
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        e, t = item
                        pts.append((float(e), float(t)))
            out[m] = pts
        return out

    # List-like input
    if isinstance(curves, list):
        if len(curves) == 0:
            return out
        # List of dicts with explicit method
        if isinstance(curves[0], dict) and all(("eps" in x and "tpr" in x) for x in curves):
            for x in curves:
                m = x.get("method", "method")
                out.setdefault(m, []).append((float(x["eps"]), float(x["tpr"])))
            return out
        # List of pairs -> unnamed single method
        if isinstance(curves[0], (list, tuple)) and len(curves[0]) == 2:
            out["method"] = [(float(e), float(t)) for (e, t) in curves]
            return out

    # Anything else -> empty
    return out

def plot_robustness(curves, out_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    data = _normalize_curves_data(curves)

    if not data:
        print("WARN: no robustness curves found (nothing to plot)")
        return []

    outputs: List[str] = []
    for method, pts in data.items():
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("adversarial ε")
        plt.ylabel("TPR")
        plt.title(f"Robustness — {method}")
        plt.grid(True, alpha=0.3)

        safe_name = method.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"robustness_{safe_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        outputs.append(out_path)

    if not outputs:
        print("WARN: robustness curves normalization produced no points")
    return outputs