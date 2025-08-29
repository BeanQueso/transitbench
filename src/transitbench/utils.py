from __future__ import annotations
import os, re, json, csv, hashlib, uuid, time, gzip, io
import numpy as np
from typing import Iterable, Dict, Any, List, Optional, Tuple

# ---------- small helpers ----------

def _param_fingerprint(d: dict) -> str:
    j = json.dumps(d, sort_keys=True, separators=(',',':'))
    return hashlib.sha1(j.encode()).hexdigest()[:12]

_slug_re = re.compile(r"[^A-Za-z0-9._-]+")
def slugify(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = _slug_re.sub("", text)
    return text or "untitled"

def ensure_dir(path: str) -> str:
    d = os.path.abspath(path)
    os.makedirs(d, exist_ok=True)
    return d

def runs_root(default: Optional[str] = None) -> str:
    """
    Resolve the canonical runs/ root.
    Override by setting env TRANSITBENCH_RUNS=/path/to/runs
    """
    root = os.environ.get("TRANSITBENCH_RUNS") or default or os.path.join(os.getcwd(), "runs")
    os.makedirs(root, exist_ok=True)
    return root

def new_run_dir(kind: str, profile: Optional[str] = None, root: Optional[str] = None, meta: Optional[dict] = None) -> str:
    """
    Create a new run directory like: runs/{kind}-{YYYYMMDD-HHMMSS}-{uuid8}[-{profile}]
    and write meta.json inside it.
    """
    root = root or runs_root()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    name = f"{kind}-{stamp}-{rid}"
    if profile:
        name = f"{name}-{slugify(profile)}"
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    write_json({"kind": kind, "created": stamp, "id": rid, "profile": profile, **(meta or {})}, os.path.join(d, "meta.json"))
    return d

def _rm_tree(path: str) -> None:
    # lightweight rm -rf
    for r, dirs, files in os.walk(path, topdown=False):
        for f in files:
            try: os.remove(os.path.join(r, f))
            except: pass
        for d in dirs:
            try: os.rmdir(os.path.join(r, d))
            except: pass
    try: os.rmdir(path)
    except: pass

def prune_runs(keep_last: int = 10, kind: Optional[str] = None, root: Optional[str] = None) -> List[str]:
    """
    Keep only the latest N run folders (by lexicographic name time order).
    If kind is given, only prune run folders starting with "kind-".
    Returns the list of deleted paths.
    """
    root = root or runs_root()
    try:
        entries = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]
    except FileNotFoundError:
        return []
    if kind:
        entries = [e for e in entries if e.startswith(f"{kind}-")]
    entries.sort()  # timestamp in name → lexical time order
    to_delete = entries[:-keep_last] if keep_last > 0 else entries
    deleted = []
    for e in to_delete:
        p = os.path.join(root, e)
        try:
            _rm_tree(p)
            deleted.append(p)
        except Exception:
            pass
    return deleted

# ---------- atomic write primitives ----------

def _atomic_write_bytes(path: str, data: bytes) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path

def write_text(text: str, path: str) -> str:
    """
    Atomically write text. If path ends with .gz, gzip it.
    """
    if path.endswith(".gz"):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(text.encode("utf-8"))
        return _atomic_write_bytes(path, buf.getvalue())
    else:
        return _atomic_write_bytes(path, text.encode("utf-8"))

def write_json(obj: Any, path: str, *, indent: int = 2) -> str:
    """
    Atomically write JSON. If path ends with .gz, gzip it.
    """
    payload = json.dumps(obj, indent=indent).encode("utf-8")
    if path.endswith(".gz"):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(payload)
        return _atomic_write_bytes(path, buf.getvalue())
    else:
        return _atomic_write_bytes(path, payload)

def append_jsonl(path: str, obj: dict):
    """
    Append a single JSON line; fsync to reduce loss on crash.
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    line = json.dumps(obj, separators=(',',':')) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())

# ---------- existing cache helpers (now atomic) ----------

def cache_path(root: str, label: str, method: str, params: dict) -> str:
    os.makedirs(root, exist_ok=True)
    fp = _param_fingerprint(params)
    base = f"{label}__{method}__{fp}.npz"
    return os.path.join(root, base)

def save_clean(cache_file: str, t: np.ndarray, f: np.ndarray, meta: dict):
    ensure_dir(os.path.dirname(os.path.abspath(cache_file)) or ".")
    tmp = f"{cache_file}.tmp.{uuid.uuid4().hex[:8]}"
    np.savez_compressed(tmp, t=t, f=f, meta=json.dumps(meta or {}))
    os.replace(tmp, cache_file)

def load_clean(cache_file: str):
    if not os.path.exists(cache_file):
        return None
    z = np.load(cache_file, allow_pickle=False)
    meta_raw = z["meta"]
    try:
        meta = json.loads(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw)
    except Exception:
        meta = {}
    return z["t"], z["f"], meta

# ---------- CSV + Markdown utilities ----------

def save_records_csv(records: Iterable[Dict[str, Any]], path: str) -> str:
    """
    Save a list of flat dicts as CSV. The header is the union of keys.
    """
    recs = list(records)
    if not recs:
        ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        open(path, "w").close()
        return path

    keys: List[str] = sorted({k for r in recs for k in r.keys()})
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in keys})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path

def write_md_summary(records: Iterable[Dict[str, Any]], path: str, title: str = "TransitBench Summary") -> str:
    """
    Simple Markdown summary of injection–recovery records.
    """
    recs = list(records)
    lines = [f"# {title}", ""]
    if not recs:
        lines.append("_No records._")
        return write_text("\n".join(lines), path)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        key = (r.get("depth"), r.get("duration"), r.get("period"), r.get("mode"))
        groups[key].append(r)

    for (depth, dur, per, mode), rows in sorted(groups.items()):
        det = sum(1 for r in rows if r.get("detected"))
        n   = len(rows)
        lines.append(f"## mode={mode} depth={depth} dur={dur} per={per} — detected {det}/{n}")
        lines.append("")
        lines.append("| detected | snr_injected | snr_top | phase_coverage | n_transits |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(f"| {int(bool(r.get('detected')))} "
                         f"| {r.get('snr_injected', 0.0):.3f} "
                         f"| {r.get('snr_top', 0.0):.3f} "
                         f"| {r.get('phase_coverage', 0.0):.2f} "
                         f"| {r.get('n_transits_observed','')} |")
        lines.append("")

    return write_text("\n".join(lines), path)

# ---------- Run context manager ----------

class Run:
    """
    Usage:
        from transitbench.utils import Run
        with Run("bench", profile="balanced", keep_last=12) as run:
            # write artifacts into this run folder
            run.write_json(res, "result.json")
            run.write_text("hello", "notes.txt")
            csv_path = run.path("records.csv")  # for functions that want a path
    """
    def __init__(self, kind: str, profile: Optional[str] = None,
                 root: Optional[str] = None, meta: Optional[dict] = None,
                 keep_last: Optional[int] = None):
        self.kind = kind
        self.profile = profile
        self.root = root
        self.meta_extra = meta or {}
        self.keep_last = keep_last
        self.dir: str = ""

    def __enter__(self) -> "Run":
        self.dir = new_run_dir(self.kind, self.profile, self.root, meta=self.meta_extra)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.keep_last is not None:
            prune_runs(keep_last=self.keep_last, kind=self.kind, root=self.root)
        # don't suppress exceptions
        return False

    # handy path + write helpers
    def path(self, *parts: str) -> str:
        p = os.path.join(self.dir, *parts)
        ensure_dir(os.path.dirname(p) or ".")
        return p

    def write_json(self, obj: Any, *parts: str, indent: int = 2) -> str:
        return write_json(obj, self.path(*parts), indent=indent)

    def write_text(self, text: str, *parts: str) -> str:
        return write_text(text, self.path(*parts))

    def append_jsonl(self, obj: dict, *parts: str) -> None:
        return append_jsonl(self.path(*parts), obj)