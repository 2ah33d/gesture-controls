"""Camera capture module for GestureFlow.

Provides a minimal, fast wrapper around OpenCV's VideoCapture
optimised for Windows (DirectShow backend).
"""

import platform

import cv2
import numpy as np


class CameraCapture:
    """Thin wrapper around cv2.VideoCapture for low-latency frame grabbing."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """Open a camera and configure the requested resolution.

        Parameters
        ----------
        camera_index:
            Index of the video-capture device (0 = default webcam).
        width:
            Desired frame width in pixels.
        height:
            Desired frame height in pixels.

        Raises
        ------
        RuntimeError
            If the camera cannot be opened.
        """
        # Use DirectShow backend on Windows for lower latency.
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(camera_index, backend)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {camera_index} "
                f"(backend={'CAP_DSHOW' if backend == cv2.CAP_DSHOW else 'CAP_ANY'})"
            )

        # Request the desired resolution; the camera may negotiate a
        # different one – actual values are exposed via properties.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab a single frame from the camera.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            ``(True, frame)`` on success, ``(False, None)`` on failure.
            The frame is the raw BGR numpy array – no copies are made.
        """
        success, frame = self._cap.read()
        if not success:
            return False, None
        return True, frame

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_width(self) -> int:
        """Actual frame width negotiated with the camera."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Actual frame height negotiated with the camera."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
