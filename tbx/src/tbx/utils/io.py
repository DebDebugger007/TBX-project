"""Small IO helpers shared across the scripts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def timestamp() -> str:
    """UTC timestamp for run folders, e.g. 20260312T140501Z."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(o: Any):
    # numpy scalars / arrays turn up in metric dicts; make them JSON-safe.
    try:
        import numpy as np

        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
