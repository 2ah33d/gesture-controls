"""Gesture engine: detects clicks, right-clicks, and scroll from hand landmarks.

Gesture recognition is based on normalised inter-landmark distances with
hysteresis to avoid flickering at threshold boundaries. Includes debouncing
to prevent single-frame noise spikes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

@dataclass
class GestureState:
    """Snapshot of the gesture state for a single frame."""

    left_down: bool = False
    left_changed: bool = False
    right_down: bool = False
    right_changed: bool = False
    scroll_active: bool = False
    scroll_delta: float = 0.0


class GestureEngine:
    """Stateful gesture recogniser operating on MediaPipe hand landmarks.

    Landmarks are expected as a list-like of objects with ``.x`` and ``.y``
    attributes (normalised 0-1), matching the MediaPipe NormalizedLandmark
    format.

    Supported gestures
    ------------------
    * **Left click** – thumb-tip (4) to index-tip (8) pinch.
    * **Right click** – thumb-tip (4) to middle-tip (12) pinch, suppressed
      while left click is active.
    * **Scroll** – index (8) and middle (12) fingers extended while ring (16)
      and pinky (20) are curled.  Vertical delta is tracked frame-to-frame.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.12,
        release_threshold: float = 0.18,
        scroll_sensitivity: float = 5.0,
        scroll_natural: bool = True,
        debounce_frames: int = 2,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self.release_threshold = release_threshold
        self.scroll_sensitivity = scroll_sensitivity
        self.scroll_natural = scroll_natural
        self.debounce_frames = debounce_frames

        # Previous frame state.
        self._prev_left_down: bool = False
        self._prev_right_down: bool = False
        self._prev_scroll_active: bool = False
        self._prev_scroll_y: float | None = None

        # Debounce counters
        self._left_pinch_count: int = 0
        self._left_release_count: int = 0
        self._right_pinch_count: int = 0
        self._right_release_count: int = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _get_extended_tip(landmarks, tip_idx: int, dip_idx: int, extension: float = 0.25) -> Point:
        """Projects a point further along the line from the DIP joint to the TIP.
        This compensates for MediaPipe placing the tip landmark slightly below
        the actual physical end of the user's finger.
        """
        tip = landmarks[tip_idx]
        dip = landmarks[dip_idx]
        dx = tip.x - dip.x
        dy = tip.y - dip.y
        return Point(tip.x + dx * extension, tip.y + dy * extension)

    @staticmethod
    def _normalized_distance(
        pt_a,
        pt_b,
        landmarks,
        ref_idx_a: int = 5,
        ref_idx_b: int = 17,
    ) -> float:
        """Euclidean distance between two arbitrary points, normalised by the
        reference distance (default: index MCP to pinky MCP = palm width).
        """
        dx = pt_a.x - pt_b.x
        dy = pt_a.y - pt_b.y
        raw = math.hypot(dx, dy)

        # Normalise by palm width
        ref_a = landmarks[ref_idx_a]
        ref_b = landmarks[ref_idx_b]
        ref = math.hypot(ref_a.x - ref_b.x, ref_a.y - ref_b.y)
        if ref < 1e-6:
            return 0.0
        return raw / ref

    @staticmethod
    def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
        """Return *True* if the fingertip is above (lower y) the PIP joint.

        In normalised camera coordinates y increases downward, so an
        extended finger has ``tip.y < pip.y``.
        """
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    # -- public API -------------------------------------------------------

    def update(self, landmarks) -> GestureState:
        """Process a single frame of hand landmarks and return gesture state.

        Parameters
        ----------
        landmarks:
            Sequence of 21 hand landmarks (e.g. ``hand_landmarks.landmark``).

        Returns
        -------
        GestureState
        """
        state = GestureState()

        # Compute extended tips to bridge the physical vs camera gap
        thumb_ext = self._get_extended_tip(landmarks, 4, 3, extension=0.25)
        index_ext = self._get_extended_tip(landmarks, 8, 7, extension=0.25)
        middle_ext = self._get_extended_tip(landmarks, 12, 11, extension=0.25)

        # ---- Left click: thumb(4) – index(8) pinch ----
        dist_left = self._normalized_distance(thumb_ext, index_ext, landmarks)
        
        target_left_down = self._prev_left_down
        if self._prev_left_down:
            if dist_left > self.release_threshold:
                self._left_release_count += 1
                if self._left_release_count >= self.debounce_frames:
                    target_left_down = False
            else:
                self._left_release_count = 0
        else:
            if dist_left < self.pinch_threshold:
                self._left_pinch_count += 1
                if self._left_pinch_count >= self.debounce_frames:
                    target_left_down = True
            else:
                self._left_pinch_count = 0

        state.left_down = target_left_down
        state.left_changed = state.left_down != self._prev_left_down

        # ---- Right click: thumb(4) – middle(12) pinch ----
        dist_right = self._normalized_distance(thumb_ext, middle_ext, landmarks)
        
        target_right_down = self._prev_right_down
        if self._prev_right_down:
            if dist_right > self.release_threshold:
                self._right_release_count += 1
                if self._right_release_count >= self.debounce_frames:
                    target_right_down = False
            else:
                self._right_release_count = 0
        else:
            if dist_right < self.pinch_threshold:
                self._right_pinch_count += 1
                if self._right_pinch_count >= self.debounce_frames:
                    target_right_down = True
            else:
                self._right_pinch_count = 0

        state.right_down = target_right_down

        # Suppress right click while left click is active.
        if state.left_down:
            state.right_down = False
            
        state.right_changed = state.right_down != self._prev_right_down

        # ---- Scroll: index + middle extended, ring + pinky curled ----
        index_extended = self._is_finger_extended(landmarks, 8, 6)
        middle_extended = self._is_finger_extended(landmarks, 12, 10)
        ring_curled = not self._is_finger_extended(landmarks, 16, 14)
        pinky_curled = not self._is_finger_extended(landmarks, 20, 18)

        state.scroll_active = (
            index_extended and middle_extended and ring_curled and pinky_curled
        )

        if state.scroll_active:
            # Use the midpoint of index and middle tips for vertical tracking.
            scroll_y = (landmarks[8].y + landmarks[12].y) / 2.0
            if self._prev_scroll_active and self._prev_scroll_y is not None:
                dy = scroll_y - self._prev_scroll_y
                direction = 1.0 if self.scroll_natural else -1.0
                state.scroll_delta = dy * self.scroll_sensitivity * direction
            else:
                state.scroll_delta = 0.0
            self._prev_scroll_y = scroll_y
        else:
            state.scroll_delta = 0.0
            self._prev_scroll_y = None

        # ---- Store state for next frame ----
        self._prev_left_down = state.left_down
        self._prev_right_down = state.right_down
        self._prev_scroll_active = state.scroll_active

        return state
