from __future__ import annotations
import os
import re

_TIC_RE = re.compile(r"(?:^|[^A-Z0-9])TIC[_\-]?(?P<num>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_TOI_RE = re.compile(r"(?:^|[^A-Z0-9])TOI[_\-]?(?P<num>\d+(?:\.\d+)?)(?:[^0-9]|$)", re.IGNORECASE)

def parse_target_label(path: str) -> str:
    """
    Try to extract TOI or TIC from filename or nearby parent folders.
    Preference: TOI > TIC. Fallback: base filename (sans extension).
    """
    parts = []
    p = os.path.abspath(path)
    for _ in range(4):  # filename + up to 3 parents
        parts.append(os.path.basename(p))
        p = os.path.dirname(p)
    text = " ".join(parts)

    m_toi = _TOI_RE.search(text)
    if m_toi:
        return f"TOI_{m_toi.group('num')}".replace("__", "_")

    m_tic = _TIC_RE.search(text)
    if m_tic:
        return f"TIC_{m_tic.group('num')}".replace("__", "_")

    return os.path.splitext(os.path.basename(path))[0]
