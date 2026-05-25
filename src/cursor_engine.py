"""Cursor engine: maps normalized hand landmarks to screen coordinates.

Uses a virtual bounding box and 1€ filtering for smooth, low-latency cursor
movement, with support for dead zones and x-axis mirroring.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# 1€ Filter – pure-Python implementation
# Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
# for Noisy Input in Interactive Systems", CHI 2012.
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """Speed-adaptive low-pass filter (1€ filter)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    @staticmethod
    def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
        return a * x + (1.0 - a) * x_prev

    # -- public API -------------------------------------------------------

    def __call__(self, x: float, timestamp: float) -> float:
        """Filter a single scalar value at *timestamp* (seconds)."""
        if self._t_prev is None:
            # First sample – no filtering possible yet.
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return x

        t_e = timestamp - self._t_prev
        if t_e <= 0:
            # Guard against duplicate / out-of-order timestamps.
            return self._x_prev  # type: ignore[return-value]

        # Derivative estimation.
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e  # type: ignore[operator]
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # Adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filtered signal.
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev)  # type: ignore[arg-type]

        # Store state.
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp

        return x_hat


# ---------------------------------------------------------------------------
# Cursor Engine
# ---------------------------------------------------------------------------

class CursorEngine:
    """Maps normalised hand-landmark coordinates to screen-pixel positions.

    A configurable *virtual box* (fraction of the camera frame) is used as
    the active region.  Landmarks inside the box are linearly mapped to
    the full screen area; movements outside the box are ignored (the cursor
    holds its last known position).

    Two independent :class:`OneEuroFilter` instances smooth the X and Y
    axes separately.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        box_width_frac: float,
        box_height_frac: float,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
        dead_zone_px: int = 3,
        mirror_x: bool = True,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Virtual-box boundaries (normalised 0-1 space).
        self.box_left = (1.0 - box_width_frac) / 2.0
        self.box_top = (1.0 - box_height_frac) / 2.0
        self.box_width = box_width_frac
        self.box_height = box_height_frac

        self.dead_zone_px = dead_zone_px
        self.mirror_x = mirror_x

        # One filter per axis.
        self._filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

        # Last known good screen position.
        self._last_x: int = screen_width // 2
        self._last_y: int = screen_height // 2

        # Freeze state.
        self._frozen = False
        self._frozen_x: int = self._last_x
        self._frozen_y: int = self._last_y

    # -- public API -------------------------------------------------------

    def update(
        self,
        landmark_x: float,
        landmark_y: float,
        timestamp: float,
        is_clicking: bool = False,
    ) -> tuple[int, int, bool]:
        """Compute screen-space cursor position from a normalised landmark.

        Parameters
        ----------
        landmark_x, landmark_y:
            Hand-landmark position in normalised [0, 1] coordinates.
        timestamp:
            Current time in seconds (monotonic preferred).
        is_clicking:
            Reserved for future click-specific behaviour.

        Returns
        -------
        (screen_x, screen_y, in_bounds)
            Pixel coordinates clamped to the screen, and whether the
            landmark fell inside the virtual box.
        """
        if self._frozen:
            return self._frozen_x, self._frozen_y, True

        # Mirror x-axis if requested (because unflipped frame has 0 on the left, but that's user right)
        if self.mirror_x:
            landmark_x = 1.0 - landmark_x

        # Check if the landmark is inside the virtual box.
        in_bounds = (
            self.box_left <= landmark_x <= self.box_left + self.box_width
            and self.box_top <= landmark_y <= self.box_top + self.box_height
        )

        if not in_bounds:
            return self._last_x, self._last_y, False

        # Map from virtual-box normalised coords → [0, 1].
        norm_x = (landmark_x - self.box_left) / self.box_width
        norm_y = (landmark_y - self.box_top) / self.box_height

        # Map to screen pixels.
        raw_x = norm_x * self.screen_width
        raw_y = norm_y * self.screen_height

        # Smooth with 1€ filter.
        smooth_x = self._filter_x(raw_x, timestamp)
        smooth_y = self._filter_y(raw_y, timestamp)

        # Clamp to screen bounds.
        screen_x = int(max(0, min(self.screen_width - 1, smooth_x)))
        screen_y = int(max(0, min(self.screen_height - 1, smooth_y)))

        # Dead zone check to eliminate micro-jitter
        dx = abs(screen_x - self._last_x)
        dy = abs(screen_y - self._last_y)
        if dx < self.dead_zone_px and dy < self.dead_zone_px:
            return self._last_x, self._last_y, True

        self._last_x = screen_x
        self._last_y = screen_y

        return screen_x, screen_y, True

    def freeze(self) -> None:
        """Freeze the cursor at its current position."""
        self._frozen = True
        self._frozen_x = self._last_x
        self._frozen_y = self._last_y

    def unfreeze(self) -> None:
        """Resume normal cursor tracking."""
        self._frozen = False

    def get_virtual_box(self) -> tuple[float, float, float, float]:
        """Return the virtual box as *(left, top, width, height)* in normalised coords."""
        return self.box_left, self.box_top, self.box_width, self.box_height
