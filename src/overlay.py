"""Debug overlay for drawing hand tracking visualizations on the webcam frame."""

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

# Get hand connections as (start, end) tuples
HAND_CONNECTIONS = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]


# Finger landmark index ranges for coloring
_FINGER_COLORS = {
    # Thumb: landmarks 1-4
    "thumb": ((0, 1, 2, 3, 4), (0, 0, 255)),       # Red (BGR)
    # Index: landmarks 5-8
    "index": ((5, 6, 7, 8), (0, 255, 0)),           # Green
    # Middle: landmarks 9-12
    "middle": ((9, 10, 11, 12), (255, 0, 0)),       # Blue
    # Ring: landmarks 13-16
    "ring": ((13, 14, 15, 16), (0, 255, 255)),      # Yellow
    # Pinky: landmarks 17-20
    "pinky": ((17, 18, 19, 20), (255, 0, 255)),     # Purple
}

# Build a lookup: landmark_index -> color
_LANDMARK_COLOR_MAP = {}
for _finger, (_indices, _color) in _FINGER_COLORS.items():
    for _idx in _indices:
        _LANDMARK_COLOR_MAP[_idx] = _color
# Wrist (landmark 0) gets white
_LANDMARK_COLOR_MAP[0] = (255, 255, 255)

# Connection color: use the color of the endpoint with the higher index
_CONNECTION_COLOR_MAP = {}
for _conn in HAND_CONNECTIONS:
    _max_idx = max(_conn)
    _CONNECTION_COLOR_MAP[_conn] = _LANDMARK_COLOR_MAP.get(_max_idx, (200, 200, 200))


class DebugOverlay:
    """Draws debug information directly onto the webcam frame."""

    def draw(
        self,
        frame: np.ndarray,
        landmarks=None,
        virtual_box=None,
        screen_pos=None,
        gesture_state=None,
        fps: float = 0.0,
        is_active: bool = True,
        tracking_landmark: str = "palm_center",
    ) -> np.ndarray:
        """Draw all debug overlays onto *frame* (mutated in-place) and return it.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the webcam.
        landmarks : list or None
            List of 21 MediaPipe NormalizedLandmark objects (.x, .y in 0-1).
        virtual_box : tuple or None
            (left, top, width, height) as fractions of the frame size.
        screen_pos : tuple or None
            (screen_x, screen_y) — the mapped cursor position on screen.
        gesture_state : GestureState or None
            Current gesture state object with .left_down, .right_down, .scrolling, etc.
        fps : float
            Frames per second to display.
        is_active : bool
            Whether the controller is active (True) or paused (False).

        Returns
        -------
        np.ndarray
            The annotated frame (same object as input).
        """
        h, w = frame.shape[:2]

        # --- Virtual interaction box ---
        if virtual_box is not None:
            bx, by, bw, bh = virtual_box
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Hand landmarks ---
        if landmarks is not None:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Draw connections first (behind the dots)
            for conn in HAND_CONNECTIONS:
                i1, i2 = conn
                color = _CONNECTION_COLOR_MAP.get(conn, (200, 200, 200))
                cv2.line(frame, pts[i1], pts[i2], color, 1, cv2.LINE_AA)

            # Draw landmark dots
            for idx, pt in enumerate(pts):
                color = _LANDMARK_COLOR_MAP.get(idx, (255, 255, 255))
                cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)

            # Tracking landmark highlight
            if tracking_landmark == "palm_center":
                # Average of MCP joints 5, 9, 13, 17
                px = int((pts[5][0] + pts[9][0] + pts[13][0] + pts[17][0]) / 4.0)
                py = int((pts[5][1] + pts[9][1] + pts[13][1] + pts[17][1]) / 4.0)
                cv2.circle(frame, (px, py), 8, (0, 0, 255), -1, cv2.LINE_AA)
            else:
                # Index fingertip (landmark 8)
                cv2.circle(frame, pts[8], 8, (0, 0, 255), -1, cv2.LINE_AA)

            # Pinch visualisation
            if gesture_state is not None and getattr(gesture_state, "left_down", False):
                # Thumb tip filled green
                cv2.circle(frame, pts[4], 8, (0, 255, 0), -1, cv2.LINE_AA)
                # Line between thumb and index
                cv2.line(frame, pts[4], pts[8], (0, 255, 255), 2, cv2.LINE_AA)

            # Screen coordinates near the fingertip
            if screen_pos is not None:
                sx, sy = screen_pos
                label = f"({int(sx)}, {int(sy)})"
                cv2.putText(
                    frame,
                    label,
                    (pts[8][0] + 12, pts[8][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # --- FPS (top-left) ---
        fps_text = f"FPS: {fps:.1f}"
        # Black outline
        cv2.putText(
            frame, fps_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
        )
        # White fill
        cv2.putText(
            frame, fps_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # --- Active / Paused status (top-right) ---
        status_text = "ACTIVE" if is_active else "PAUSED"
        status_color = (0, 255, 0) if is_active else (0, 0, 255)
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame,
            status_text,
            (w - tw - 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # --- Gesture state labels (bottom) ---
        if gesture_state is not None:
            labels = []
            if getattr(gesture_state, "left_down", False):
                labels.append(("LEFT CLICK", (0, 200, 255)))   # Orange
            if getattr(gesture_state, "right_down", False):
                labels.append(("RIGHT CLICK", (255, 0, 255)))  # Magenta
            if getattr(gesture_state, "scroll_active", False):
                labels.append(("SCROLLING", (255, 255, 0)))    # Cyan

            x_offset = 10
            for text, color in labels:
                cv2.putText(
                    frame,
                    text,
                    (x_offset, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                x_offset += tw + 20

        return frame
