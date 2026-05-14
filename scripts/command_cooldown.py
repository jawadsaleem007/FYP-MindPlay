"""Shared gyro-command cooldown state helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = PROJECT_ROOT / "gamepad_state.json"


def resolve_state_file(state_file: Optional[Union[str, Path]]) -> Optional[Path]:
    """Resolve a shared state file path relative to the project root."""
    if state_file is None:
        return None

    text = str(state_file).strip()
    if not text:
        return None

    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_state(state_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load shared command state, returning an empty dict on missing/invalid files."""
    path = resolve_state_file(state_file)
    if path is None or not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cooldown_remaining_from_state(state: Dict[str, Any], now: Optional[float] = None) -> float:
    """Return remaining cooldown seconds from a loaded state dictionary."""
    current_time = time.time() if now is None else now
    cooldown_until = _float_value(state.get("cooldown_until"), 0.0)
    return max(0.0, cooldown_until - current_time)


def cooldown_status(
    state_file: Optional[Union[str, Path]],
    now: Optional[float] = None,
) -> Tuple[bool, float, str]:
    """Return whether commands are blocked, remaining seconds, and cooldown source."""
    state = load_state(state_file)
    remaining = cooldown_remaining_from_state(state, now=now)
    source = str(state.get("cooldown_source") or "")
    return remaining > 0.0, remaining, source


def cooldown_payload(cooldown_until: float, cooldown_seconds: float, cooldown_source: str) -> Dict[str, Any]:
    """Create normalized cooldown fields for the overlay state JSON."""
    if cooldown_remaining_from_state({"cooldown_until": cooldown_until}) <= 0.0:
        return {
            "cooldown_until": 0.0,
            "cooldown_seconds": 0.0,
            "cooldown_source": "",
        }

    return {
        "cooldown_until": float(cooldown_until),
        "cooldown_seconds": max(0.0, float(cooldown_seconds)),
        "cooldown_source": str(cooldown_source or ""),
    }
