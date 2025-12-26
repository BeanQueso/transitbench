import numpy as np
from typing import Dict, List, Tuple

def roc_from_scores(z_pos: np.ndarray, z_neg: np.ndarray, taus: np.ndarray) -> Dict[str, np.ndarray]:
    # z_pos: injected detections metric (e.g., max z) per trial
    # z_neg: null trials metric
    tpr = np.array([(z_pos >= t).mean() for t in taus])
    fpr = np.array([(z_neg >= t).mean() for t in taus])
    return {"tau": taus, "tpr": tpr, "fpr": fpr}

def bootstrap_ci(values: np.ndarray, fn, nboot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    idx = np.random.randint(0, len(values), size=(nboot, len(values)))
    boots = np.array([fn(values[i]) for i in idx])
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1-alpha/2))
    return lo, hi

def roc_with_ci(z_pos: np.ndarray, z_neg: np.ndarray, taus: np.ndarray, nboot: int = 1000, alpha: float = 0.05):
    tpr = []; fpr = []; tpr_lo=[]; tpr_hi=[]; fpr_lo=[]; fpr_hi=[]
    for t in taus:
        pos_hit = (z_pos >= t).astype(float)
        neg_hit = (z_neg >= t).astype(float)
        tpr.append(pos_hit.mean()); fpr.append(neg_hit.mean())
        # bootstrap CIs
        tlo, thi = bootstrap_ci(pos_hit, np.mean, nboot=nboot, alpha=alpha)
        flo, fhi = bootstrap_ci(neg_hit, np.mean, nboot=nboot, alpha=alpha)
        tpr_lo.append(tlo); tpr_hi.append(thi); fpr_lo.append(flo); fpr_hi.append(fhi)
    return {
        "tau": taus,
        "tpr": np.array(tpr), "tpr_lo": np.array(tpr_lo), "tpr_hi": np.array(tpr_hi),
        "fpr": np.array(fpr), "fpr_lo": np.array(fpr_lo), "fpr_hi": np.array(fpr_hi),
    }

def compute_completeness(records: List[dict], by=("depth","duration")) -> Dict[Tuple[float,float], float]:
    from collections import defaultdict
    bins = defaultdict(list)
    for r in records:
        key = tuple(float(r[k]) for k in by)
        bins[key].append(1.0 if r.get("detected") else 0.0)
    return {k: float(np.mean(v)) for k,v in bins.items()}
