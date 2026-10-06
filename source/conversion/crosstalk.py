"""Dye-crosstalk unmix: profile loading, resolve, process gate.

Ported from NegPy ``services/assets/crosstalk.py``, ``features/exposure/normalization.py``
(resolve/gate) and ``features/process/models.py`` (GPL-3.0, see NOTICE.md).
Bundled profiles only; no user-profile directory.
"""

import os
import tomllib
from enum import StrEnum
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

DEFAULT_NAME = "Generic C41"

# Built-in fallback crosstalk matrix (row-major 3x3).
DEFAULT_CROSSTALK_MATRIX = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)


from .modes import ProcessMode  # noqa: E402


def assets_dir() -> str:
    """Directory holding bundled assets (``crosstalk/``, ``icc/``)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, "assets")
    if os.path.isdir(local):
        return local
    try:
        from ResourcePaths import resource_path

        return resource_path("assets")
    except Exception:
        return local


def _crosstalk_dir() -> str:
    return os.path.join(assets_dir(), "crosstalk")


def _parse_file(path: str) -> Optional[dict]:
    """Parse one TOML to {name, matrix(9 floats), type, process}, or None if invalid."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        rows = data.get("matrix")
        if not isinstance(rows, list) or len(rows) != 3:
            return None
        flat: List[float] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                return None
            for v in row:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    return None
                flat.append(float(v))
        raw_name = data.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
        ptype = data.get("type")
        proc = data.get("process")
        return {
            "name": name or os.path.basename(path)[:-5],
            "matrix": flat,
            "type": ptype.strip().lower() if isinstance(ptype, str) else "",
            # A missing `process` key means colour negative.
            "process": str(ProcessMode(proc.strip() if isinstance(proc, str) else "")),
        }
    except Exception:
        return None


@lru_cache(maxsize=4)
def _scan(directory: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".toml"):
            continue
        parsed = _parse_file(os.path.join(directory, fname))
        if parsed is not None and parsed["name"] != DEFAULT_NAME:
            out[parsed["name"]] = parsed
    return out


def scan_bundled() -> Dict[str, List[float]]:
    """Bundled matrices keyed by display name."""
    return {n: p["matrix"] for n, p in _scan(_crosstalk_dir()).items()}


def list_profiles(process_mode: Optional[str] = None) -> List[str]:
    """["Generic C41", *sorted bundled names], optionally filtered to one film process."""
    profiles = _scan(_crosstalk_dir())
    names = sorted(
        n for n, p in profiles.items() if process_mode is None or p["process"] == str(process_mode)
    )
    if process_mode is None or str(process_mode) == str(ProcessMode.C41):
        names.insert(0, DEFAULT_NAME)
    return names


def get_matrix(name: str) -> Optional[List[float]]:
    """Flat 9-float list for a profile, or None for the built-in / unknown profile."""
    if name == DEFAULT_NAME:
        return None
    p = _scan(_crosstalk_dir()).get(name)
    return list(p["matrix"]) if p else None


def get_process(name: str) -> str:
    """Film process a profile was derived for; C-41 when unknown."""
    p = _scan(_crosstalk_dir()).get(name)
    return p["process"] if p else str(ProcessMode.C41)


def resolve_crosstalk_matrix(strength: float, matrix: Optional[tuple]) -> Optional[np.ndarray]:
    """Identity<->calibration blend by strength, row-normalized; None when off."""
    if float(strength) <= 0.0:
        return None
    m = matrix if matrix is not None else DEFAULT_CROSSTALK_MATRIX
    cal = np.array(m, dtype=np.float64).reshape(3, 3)
    applied = np.eye(3) * (1.0 - float(strength)) + cal * float(strength)
    row_sums = np.sum(applied, axis=1, keepdims=True)
    return applied / np.maximum(row_sums, 1e-6)


def effective_crosstalk_matrix(
    strength: float,
    matrix: Optional[tuple],
    crosstalk_process: Optional[str],
    process_mode: Optional[str],
) -> Optional[np.ndarray]:
    """Mode-aware resolve: the unmix only when the profile was derived for the film processed."""
    profile_mode = str(crosstalk_process or ProcessMode.C41)
    if process_mode is not None and profile_mode != str(process_mode):
        return None
    return resolve_crosstalk_matrix(strength, matrix)
