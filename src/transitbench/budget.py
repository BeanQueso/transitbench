from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple, Optional

def _apply_exclusions(periods: np.ndarray, exclude: Optional[List[Tuple[float,float]]]) -> np.ndarray:
    if not exclude:
        return periods
    mask = np.ones_like(periods, dtype=bool)
    for P, w in exclude:
        if not np.isfinite(P) or not np.isfinite(w) or P <= 0 or w <= 0:
            continue
        lo = P * (1.0 - w)
        hi = P * (1.0 + w)
        mask &= ~((periods >= lo) & (periods <= hi))
    kept = periods[mask]
    return kept if kept.size else periods  # never return empty

def _parse_float_list(x: Iterable[float]) -> List[float]:
    vals = [float(v) for v in x]
    # dedupe & sort
    vals = sorted({v for v in vals if np.isfinite(v) and v > 0})
    return vals

def make_budgeted_grid(
    pmin: float,
    pmax: float,
    durations: Iterable[float],
    compute_budget: int,
    exclude: Optional[List[Tuple[float,float]]] = None,
    min_period_points: int = 64,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Build (periods_grid, durations_list, total_cells) under a target compute budget.
    - pmin/pmax: period search range
    - durations: candidate durations (days)
    - compute_budget: approximate max cells (#periods * #durations)
    - exclude: [(P, frac_width), ...] fractional windows to skip
    Strategy:
      1) Start with durations as given (deduped/sorted).
      2) Pick as many period samples as fit in budget (>= min_period_points).
      3) Apply exclusions; if too many removed, top up period samples to maintain cells ~ budget.
    """
    durs = _parse_float_list(durations)
    if compute_budget is None or compute_budget <= 0:
        # no budget: return a default dense grid
        periods = np.linspace(pmin, pmax, 3000)
        return _apply_exclusions(periods, exclude), durs, len(durs) * 3000

    # ensure at least one duration
    if not durs:
        durs = [0.1]

    # number of period samples we can afford
    nP = max(min_period_points, compute_budget // max(1, len(durs)))
    periods = np.linspace(pmin, pmax, int(nP))
    periods = _apply_exclusions(periods, exclude)

    # If exclusions removed many, try to top up by oversampling and reapplying
    if periods.size * len(durs) < 0.7 * compute_budget:
        nP2 = int(max(nP, compute_budget // max(1, len(durs))) * 1.5)
        periods2 = np.linspace(pmin, pmax, max(nP2, periods.size + min_period_points))
        periods = _apply_exclusions(periods2, exclude)

    cells = int(periods.size) * len(durs)
    return periods, durs, cells
