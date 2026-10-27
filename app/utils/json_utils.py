"""
MangaTR-Studio - JSON Serialization & File Utilities (app/utils/json_utils.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_json_safe(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and Path objects into standard JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path | str, data: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(make_json_safe(data), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def load_json(path: Path | str) -> Any:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))
