"""Hand-tracking module for GestureFlow.

Wraps the MediaPipe Tasks HandLandmarker API to detect a single preferred
hand and return structured landmark data via the ``HandResult`` dataclass.

This uses the new Tasks API (mediapipe >= 0.10.x) instead of the deprecated
``mp.solutions.hands`` legacy API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Default model path: <project_root>/models/hand_landmarker.task
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "hand_landmarker.task",
)


class _LandmarkProxy:
    """Lightweight object matching the old NormalizedLandmark interface (.x, .y, .z).

    The Tasks API returns ``NormalizedLandmark`` from the ``landmark`` proto
    module, which already has ``.x``, ``.y``, ``.z``.  This proxy exists only
    as a safety net in case the upstream type changes shape.
    """
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


@dataclass(slots=True)
class HandResult:
    """Structured result from a single hand detection.

    Attributes
    ----------
    landmarks:
        List of 21 landmark objects with ``.x``, ``.y``, ``.z`` (normalised 0-1).
    handedness:
        ``'Left'`` or ``'Right'`` — the *actual* hand of the user.
    confidence:
        Classification confidence for the handedness label.
    """

    landmarks: list
    handedness: str
    confidence: float


class HandTracker:
    """Detect and track a single hand using the MediaPipe Tasks HandLandmarker.

    Handedness / mirroring note
    ---------------------------
    MediaPipe reports handedness assuming the input image is **not**
    mirrored.  In a typical webcam pipeline the frame is captured
    un-flipped, processed here (un-flipped), and only flipped later for
    on-screen display.  Because we feed the **un-flipped** frame to
    MediaPipe, the handedness label it returns already corresponds to the
    user's actual hand — no label inversion is needed.

    If your pipeline flips the frame *before* calling ``process()``, you
    must swap the handedness labels yourself.
    """

    def __init__(
        self,
        preferred_hand: str = "Right",
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        model_path: str | None = None,
    ) -> None:
        """Initialise the MediaPipe HandLandmarker.

        Parameters
        ----------
        preferred_hand:
            ``'Left'`` or ``'Right'`` — only this hand will be returned.
        max_num_hands:
            Maximum number of hands to detect per frame.
        min_detection_confidence:
            Minimum confidence for the initial detection to succeed.
        min_tracking_confidence:
            Minimum confidence for frame-to-frame landmark tracking.
        model_complexity:
            Ignored in the Tasks API (kept for interface compatibility).
        model_path:
            Path to the ``.task`` model file.  Defaults to
            ``<project_root>/models/hand_landmarker.task``.
        """
        self._preferred_hand = preferred_hand.capitalize()

        _model_path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.isfile(_model_path):
            raise FileNotFoundError(
                f"HandLandmarker model not found at {_model_path!r}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0  # Monotonic timestamp for VIDEO mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb) -> HandResult | None:
        """Process an RGB frame and return the preferred hand's result.

        Parameters
        ----------
        frame_rgb : np.ndarray
            An RGB image as a NumPy array (H × W × 3, ``uint8``).
            Must **not** be horizontally flipped — see class docstring.

        Returns
        -------
        HandResult | None
            Landmark data for the preferred hand, or ``None`` if no
            matching hand was detected.
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect — VIDEO mode requires a monotonically increasing timestamp
        self._frame_ts_ms += 33  # ~30 FPS
        results = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not results.hand_landmarks or not results.handedness:
            return None

        # Iterate over detected hands and pick the preferred one.
        for hand_lms, hand_info in zip(
            results.hand_landmarks,
            results.handedness,
        ):
            label: str = hand_info[0].category_name  # 'Left' or 'Right'
            confidence: float = hand_info[0].score

            if label == self._preferred_hand:
                # Convert to proxy objects with .x, .y, .z
                landmarks = [
                    _LandmarkProxy(lm.x, lm.y, lm.z) for lm in hand_lms
                ]
                return HandResult(
                    landmarks=landmarks,
                    handedness=label,
                    confidence=confidence,
                )

        return None

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
