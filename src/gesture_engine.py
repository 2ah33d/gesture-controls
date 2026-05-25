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
        pinch_threshold: float = 0.15,
        release_threshold: float = 0.22,
        scroll_sensitivity: float = 5.0,
        scroll_natural: bool = True,
        debounce_frames: int = 1,
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
        thumb_ext = self._get_extended_tip(landmarks, 4, 3, extension=0.40)
        index_ext = self._get_extended_tip(landmarks, 8, 7, extension=0.40)
        middle_ext = self._get_extended_tip(landmarks, 12, 11, extension=0.40)

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

        # ---- Scroll: index + middle extended, ring + pinky curled, thumb tucked ----
        index_extended = self._is_finger_extended(landmarks, 8, 6)
        middle_extended = self._is_finger_extended(landmarks, 12, 10)
        ring_curled = not self._is_finger_extended(landmarks, 16, 14)
        pinky_curled = not self._is_finger_extended(landmarks, 20, 18)

        # Thumb tucked: extended thumb tip is close to the palm center (red dot)
        palm_center = Point(
            (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4.0,
            (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4.0
        )
        thumb_tucked_dist = self._normalized_distance(thumb_ext, palm_center, landmarks)
        
        # 0.4 is much stricter than the previous 1.0. It requires the thumb 
        # to actually be folded inwards towards the red dot.
        thumb_tucked = thumb_tucked_dist < 0.4

        # Distance between index and middle fingers (using extended tips for accuracy)
        fingers_dist = self._normalized_distance(index_ext, middle_ext, landmarks)

        state.scroll_active = False
        state.scroll_delta = 0.0

        if index_extended and middle_extended and ring_curled and pinky_curled and thumb_tucked:
            state.scroll_active = True
            
            # Base scroll speed (approx 0.15 clicks per frame, scaled by sensitivity)
            # Default sensitivity is 5.0, so this gives ~0.75 clicks/frame (22 clicks/sec)
            # We'll reduce the base factor so default sensitivity gives a nice reading speed.
            base_speed = 0.08 * self.scroll_sensitivity 
            direction = 1.0 if self.scroll_natural else -1.0
            
            if fingers_dist > 0.7:  # Wide peace sign -> scroll down
                state.scroll_delta = -base_speed * direction
            elif fingers_dist < 0.45:  # Fingers close -> scroll up
                state.scroll_delta = base_speed * direction

        # ---- Store state for next frame ----
        self._prev_left_down = state.left_down
        self._prev_right_down = state.right_down
        self._prev_scroll_active = state.scroll_active

        return state
