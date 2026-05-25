"""GestureFlow — main entry point.

Wires together camera capture, hand tracking, gesture recognition,
cursor mapping, input dispatch, and debug overlay into a single loop.
"""

import sys
import os
import time

# Ensure the project root is on the path so config.ini / sibling modules resolve.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SRC_DIR)

import cv2
from screeninfo import get_monitors

from config import Config
from capture import CameraCapture
from hand_tracker import HandTracker
from cursor_engine import CursorEngine
from gesture_engine import GestureEngine, GestureState
from input_dispatch import InputDispatcher
from overlay import DebugOverlay


def main() -> None:
    # ── Configuration ────────────────────────────────────────────────
    config_path = os.path.join(_PROJECT_ROOT, "config.ini")
    cfg = Config(config_path)

    # ── Screen resolution ────────────────────────────────────────────
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height

    # ── Module initialisation ────────────────────────────────────────
    camera = CameraCapture(camera_index=0, width=1280, height=720)

    tracker = HandTracker(
        preferred_hand=cfg.preferred_hand.capitalize(),
        max_num_hands=1,
        min_detection_confidence=cfg.confidence_min,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    cursor = CursorEngine(
        screen_width=screen_w,
        screen_height=screen_h,
        box_width_frac=cfg.box_width,
        box_height_frac=cfg.box_height,
        min_cutoff=cfg.min_cutoff,
        beta=cfg.beta,
        d_cutoff=cfg.d_cutoff,
        dead_zone_px=cfg.dead_zone_px,
        mirror_x=cfg.mirror_x,
    )

    gestures = GestureEngine(
        pinch_threshold=cfg.pinch_threshold,
        release_threshold=cfg.release_threshold,
        scroll_sensitivity=cfg.scroll_sensitivity,
        scroll_natural=cfg.scroll_natural,
        debounce_frames=cfg.debounce_frames,
    )

    dispatcher = InputDispatcher()
    overlay = DebugOverlay()

    debug_mode: bool = cfg.debug_enabled
    is_active: bool = True

    # ── FPS tracking ─────────────────────────────────────────────────
    prev_time = time.perf_counter()
    fps: float = 0.0

    print(f"[GestureFlow] Starting — screen {screen_w}x{screen_h}, "
          f"debug={'on' if debug_mode else 'off'}")
    print("[GestureFlow] Press 'g' to toggle active/paused, 'q' or ESC to quit.")

    try:
        while True:
            # ── 1. Capture frame ─────────────────────────────────────
            success, frame = camera.read()
            if not success or frame is None:
                continue

            now = time.perf_counter()

            # ── 2. Convert to RGB for MediaPipe (process un-flipped
            #       so handedness is correct) ─────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── 3. Hand tracking ─────────────────────────────────────
            result = tracker.process(rgb_frame)

            landmarks = None
            gesture_state: GestureState | None = None
            screen_pos: tuple[int, int] | None = None

            if result is not None and result.landmarks:
                landmarks = result.landmarks

                # ── 4. Gesture recognition ───────────────────────────
                gesture_state = gestures.update(landmarks)

                # ── 5. Cursor mapping ────────────────────────────────
                if cfg.tracking_landmark == "palm_center":
                    # Average of MCP joints: 5, 9, 13, 17
                    track_x = (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4.0
                    track_y = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4.0
                else:
                    # Default to index tip
                    track_x = landmarks[8].x
                    track_y = landmarks[8].y

                screen_x, screen_y, in_bounds = cursor.update(
                    track_x, track_y, now
                )
                screen_pos = (screen_x, screen_y)

                # ── 6. Input dispatch (only when active) ─────────────
                if is_active:
                    # Always move cursor
                    dispatcher.move_to(screen_x, screen_y)

                    # Left click edges
                    if gesture_state.left_changed:
                        if gesture_state.left_down:
                            dispatcher.mouse_down("left")
                        else:
                            dispatcher.mouse_up("left")

                    # Right click edges
                    if gesture_state.right_changed:
                        if gesture_state.right_down:
                            dispatcher.mouse_down("right")
                        else:
                            dispatcher.mouse_up("right")

                    # Scroll
                    if gesture_state.scroll_active:
                        delta = int(gesture_state.scroll_delta)
                        if delta != 0:
                            dispatcher.scroll(delta)

            # ── 7. FPS calculation ───────────────────────────────────
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            # ── 8. Debug display ─────────────────────────────────────
            if debug_mode:
                # Flip frame for mirror display
                display_frame = cv2.flip(frame, 1)

                # Mirror landmark x-coordinates for the flipped display frame
                mirrored_landmarks = None
                if landmarks is not None:
                    # Create lightweight wrapper that mirrors .x
                    class MirroredLandmark:
                        __slots__ = ('x', 'y', 'z')
                        def __init__(self, lm):
                            self.x = 1.0 - lm.x  # Mirror horizontally
                            self.y = lm.y
                            self.z = lm.z if hasattr(lm, 'z') else 0.0
                    mirrored_landmarks = [MirroredLandmark(lm) for lm in landmarks]

                overlay.draw(
                    display_frame,
                    landmarks=mirrored_landmarks,
                    virtual_box=cursor.get_virtual_box(),
                    screen_pos=screen_pos,
                    gesture_state=gesture_state,
                    fps=fps,
                    is_active=is_active,
                    tracking_landmark=cfg.tracking_landmark,
                )
                cv2.imshow("GestureFlow", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    break
                
                # Check if user clicked the 'X' button to close the window
                if cv2.getWindowProperty("GestureFlow", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("g"):  # toggle active/paused
                    is_active = not is_active
                    print(f"[GestureFlow] {'ACTIVE' if is_active else 'PAUSED'}")
            else:
                # Still need a tiny delay and exit mechanism without GUI
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[GestureFlow] Interrupted by user.")
    finally:
        tracker.release()
        camera.release()
        cv2.destroyAllWindows()
        print("[GestureFlow] Shutdown complete.")


if __name__ == "__main__":
    main()
