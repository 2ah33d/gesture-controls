"""
GestureFlow – Configuration loader.

Reads ``config.ini`` (expected next to the project root) and exposes every
setting as a typed Python attribute.  Falls back to sensible defaults when
the file or individual keys are missing.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Virtual-box size presets  (width_fraction, height_fraction)
# ---------------------------------------------------------------------------
BOX_PRESETS: Dict[str, Tuple[float, float]] = {
    "small":  (0.35, 0.30),
    "medium": (0.55, 0.45),
    "large":  (0.75, 0.65),
}

# Default path: <project_root>/config.ini  (src/ is one level below root)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.ini"


class Config:
    """Centralised, typed access to every GestureFlow setting.

    Parameters
    ----------
    path : str | Path | None
        Explicit path to a ``config.ini`` file.  When *None* the loader
        looks for ``config.ini`` in the project root (one directory above
        ``src/``).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._parser = configparser.ConfigParser()

        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        # read() silently returns an empty list when the file is missing,
        # so all attributes will simply get their defaults.
        self._parser.read(str(config_path), encoding="utf-8")

        # ── virtual_box ───────────────────────────────────────────────
        self.box_preset: str = self._get("virtual_box", "preset", "medium")
        self.custom_width: float = self._getfloat("virtual_box", "custom_width", 0.55)
        self.custom_height: float = self._getfloat("virtual_box", "custom_height", 0.45)
        self.box_width: float
        self.box_height: float
        self.box_width, self.box_height = self._resolve_box_dimensions()

        # ── hand ──────────────────────────────────────────────────────
        self.preferred_hand: str = self._get("hand", "preferred", "right")

        # ── thresholds ────────────────────────────────────────────────
        self.pinch_threshold: float = self._getfloat("thresholds", "pinch_threshold", 0.15)
        self.release_threshold: float = self._getfloat("thresholds", "release_threshold", 0.22)
        self.confidence_min: float = self._getfloat("thresholds", "confidence_min", 0.7)
        self.debounce_frames: int = self._getint("thresholds", "debounce_frames", 1)

        # ── smoothing (1€ filter parameters) ──────────────────────────
        self.min_cutoff: float = self._getfloat("smoothing", "min_cutoff", 0.4)
        self.beta: float = self._getfloat("smoothing", "beta", 0.7)
        self.d_cutoff: float = self._getfloat("smoothing", "d_cutoff", 1.0)
        self.dead_zone_px: int = self._getint("smoothing", "dead_zone_px", 3)

        # ── cursor ────────────────────────────────────────────────────
        self.tracking_landmark: str = self._get("cursor", "tracking_landmark", "palm_center")
        self.mirror_x: bool = self._getbool("cursor", "mirror_x", True)

        # ── scroll ────────────────────────────────────────────────────
        self.scroll_sensitivity: float = self._getfloat("scroll", "sensitivity", 5.0)
        self.scroll_natural: bool = self._getbool("scroll", "natural", True)

        # ── toggle ────────────────────────────────────────────────────
        self.toggle_hotkey: str = self._get("toggle", "hotkey", "ctrl+alt+g")

        # ── debug ─────────────────────────────────────────────────────
        self.debug_enabled: bool = self._getbool("debug", "enabled", True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get(self, section: str, key: str, fallback: str) -> str:
        return self._parser.get(section, key, fallback=fallback)

    def _getfloat(self, section: str, key: str, fallback: float) -> float:
        return self._parser.getfloat(section, key, fallback=fallback)

    def _getint(self, section: str, key: str, fallback: int) -> int:
        return self._parser.getint(section, key, fallback=fallback)

    def _getbool(self, section: str, key: str, fallback: bool) -> bool:
        return self._parser.getboolean(section, key, fallback=fallback)

    def _resolve_box_dimensions(self) -> Tuple[float, float]:
        """Return *(width, height)* fractions for the virtual control box.

        If *preset* matches a key in :data:`BOX_PRESETS`, that pair is used.
        Otherwise (including ``"custom"``), the explicit *custom_width* and
        *custom_height* values are returned.
        """
        if self.box_preset in BOX_PRESETS:
            return BOX_PRESETS[self.box_preset]
        return (self.custom_width, self.custom_height)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(attrs.items()))
        return f"Config({inner})"
