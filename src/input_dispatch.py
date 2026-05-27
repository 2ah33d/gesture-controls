"""Input dispatcher: thin wrapper around pynput for mouse control.

All mouse interaction funnels through this module so the rest of the
application never touches pynput directly.
"""

from __future__ import annotations

from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as KeyboardController, Key


class InputDispatcher:
    """Dead-simple mouse-input facade."""

    _BUTTON_MAP = {
        "left": Button.left,
        "right": Button.right,
    }

    def __init__(self) -> None:
        self._mouse = Controller()
        self._keyboard = KeyboardController()

    def move_to(self, x: int, y: int) -> None:
        """Set the absolute cursor position."""
        self._mouse.position = (x, y)

    def mouse_down(self, button: str = "left") -> None:
        """Press a mouse button."""
        self._mouse.press(self._BUTTON_MAP[button])

    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button."""
        self._mouse.release(self._BUTTON_MAP[button])

    def scroll(self, clicks: int) -> None:
        """Scroll vertically by *clicks* (positive = up, negative = down)."""
        self._mouse.scroll(0, clicks)

    def zoom_in(self) -> None:
        """Send Ctrl+= (zoom in — universal shortcut for browsers, Office, OneNote)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("=")
            self._keyboard.release("=")

    def zoom_out(self) -> None:
        """Send Ctrl+- (zoom out)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("-")
            self._keyboard.release("-")
