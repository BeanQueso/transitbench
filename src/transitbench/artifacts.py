# transitbench/artifacts.py
from __future__ import annotations
import json, os, sys, gzip, uuid, shutil, time, tempfile, subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return None

def runs_root() -> Path:
    # Allow override with env var; default to <repo>/runs
    return Path(os.getenv("TRANSITBENCH_RUNS", "runs")).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    _ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)  # atomic on POSIX

def write_json(path: Path, obj: Any, compress: bool = False) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    if compress or path.suffix == ".gz":
        if path.suffix != ".gz":
            path = path.with_suffix(path.suffix + ".gz")
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
            with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
                gz.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    else:
        _atomic_write_bytes(path, data)

def write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))

def append_jsonl(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    line = json.dumps(obj, sort_keys=True)
    # append is not atomic; write to temp file then cat+replace
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    with open(tmp, "w") as fh:
        fh.write(line + "\n")
    if path.exists():
        # merge old + new
        merged = path.with_suffix(path.suffix + f".merge-{uuid.uuid4().hex}")
        with open(merged, "wb") as out:
            with open(path, "rb") as old:
                shutil.copyfileobj(old, out)
            with open(tmp, "rb") as new:
                shutil.copyfileobj(new, out)
        os.replace(merged, path)
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, path)

try:
    import numpy as np
except Exception:
    np = None

def write_numpy_npz(path: Path, **arrays) -> None:
    if np is None:
        raise RuntimeError("NumPy not available")
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

def write_dataframe(df, path: Path, fmt: str = "parquet") -> None:
    _ensure_dir(path.parent)
    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported DataFrame format: {fmt}")

def save_figure(fig, path: Path, dpi: int = 150) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    os.replace(tmp, path)

@dataclass
class RunMeta:
    run_id: str
    kind: str                      # e.g. "null_audit", "completeness", "profiles", "report"
    profile: Optional[str] = None  # e.g. "balanced"
    created_at: str = field(default_factory=_now_iso)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    git_sha: Optional[str] = field(default_factory=_git_sha)
    tb_version: Optional[str] = None
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

class Run:
    """
    One place to create folders and write artifacts.
    Use like:

        with Run(kind="completeness", profile="balanced", config=cfg) as run:
            run.save_json("completeness_balanced.json", results)
            run.finalize(summary={"global_tpr": 0.976, "n": 3240})
    """
    def __init__(self, kind: str, profile: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        rid = f"{kind}-{_now_iso()}-{uuid.uuid4().hex[:8]}"
        self.root = runs_root()
        self.path = self.root / rid
        self.meta = RunMeta(run_id=rid, kind=kind, profile=profile,
                            seed=seed, config=config or {})
        _ensure_dir(self.path)

        # write meta immediately
        write_json(self.path / "meta.json", asdict(self.meta))

    def save_json(self, name: str, obj: Any, compress: bool = False) -> Path:
        p = self.path / name
        write_json(p, obj, compress=compress)
        return p

    def save_text(self, name: str, text: str) -> Path:
        p = self.path / name
        write_text(p, text)
        return p

    def save_npz(self, name: str, **arrays) -> Path:
        p = self.path / name
        write_numpy_npz(p, **arrays)
        return p

    def save_df(self, df, name: str, fmt: str = "parquet") -> Path:
        p = self.path / name
        write_dataframe(df, p, fmt=fmt)
        return p

    def save_fig(self, fig, name: str, dpi: int = 150) -> Path:
        p = self.path / name
        save_figure(fig, p, dpi=dpi)
        return p

    def log(self, event: Dict[str, Any]) -> None:
        event = {"ts": _now_iso(), **event, "run_id": self.meta.run_id}
        append_jsonl(self.root / "index.jsonl", event)

    def finalize(self, summary: Dict[str, Any]) -> None:
        # Write a summary file into the run folder and append to global index.
        write_json(self.path / "summary.json", summary)
        idx_record = {
            "run_id": self.meta.run_id,
            "kind": self.meta.kind,
            "profile": self.meta.profile,
            "created_at": self.meta.created_at,
            "summary": summary,
        }
        append_jsonl(self.root / "index.jsonl", idx_record)

    # Optional: keep the last N runs per kind to control clutter
    @staticmethod
    def prune(keep_last: int = 10, kind: Optional[str] = None) -> None:
        root = runs_root()
        if not root.exists():
            return
        runs = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            meta = p / "meta.json"
            if not meta.exists():
                continue
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            if (kind is None) or (m.get("kind") == kind):
                runs.append((p, m.get("created_at", ""), m.get("run_id", "")))
        # newest first
        runs.sort(key=lambda t: t[1], reverse=True)
        for p, _, _ in runs[keep_last:]:
            shutil.rmtree(p, ignore_errors=True)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb):
        # Could write an error marker if needed
        if exc_type is not None:
            self.log({"level": "ERROR", "message": str(exc)})
        return False
