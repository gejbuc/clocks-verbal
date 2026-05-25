"""
Transition — Fade-to-Black Overlay

Creates a borderless, always-on-top fullscreen window that fades to black
and back, masking the hard window swap underneath.

The overlay runs on its own thread so the caller just does:

    fade = FadeTransition(duration=0.4, color=(0, 0, 0))
    fade.run(swap_fn)          # blocks until the full fade-in/swap/fade-out is done

`swap_fn` is called at peak opacity (screen fully black) — that's the moment
the underlying windows are rearranged, completely hidden from the viewer.

Dependencies: tkinter (stdlib), no extra installs needed.
"""

import threading
import tkinter as tk
from typing import Callable, Tuple


class FadeTransition:
    """
    Fullscreen fade-to-colour overlay that masks a window swap.

    Args:
        duration:   Total duration of the full transition in seconds.
                    Split evenly between fade-in and fade-out (each = duration / 2).
        color:      RGB tuple for the overlay colour. Default is black (0, 0, 0).
        steps:      Number of opacity steps per fade direction.
                    More steps = smoother but slightly more CPU.
    """

    def __init__(
        self,
        duration: float = 0.5,
        color: Tuple[int, int, int] = (0, 0, 0),
        steps: int = 20,
    ):
        self.duration = duration
        self.color    = color
        self.steps    = steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, swap_fn: Callable[[], None]) -> None:
        """
        Block until the full transition completes:
          1. Fade in (transparent → opaque)
          2. Call swap_fn() at peak opacity
          3. Fade out (opaque → transparent)
          4. Destroy the overlay window

        Args:
            swap_fn: Zero-argument callable executed at peak opacity.
                     This is where you minimize/focus windows.
        """
        done = threading.Event()
        threading.Thread(
            target=self._run_tk,
            args=(swap_fn, done),
            daemon=True,
        ).start()
        done.wait()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hex_color(self) -> str:
        r, g, b = self.color
        return f"#{r:02x}{g:02x}{b:02x}"

    def _run_tk(self, swap_fn: Callable[[], None], done: threading.Event) -> None:
        """Runs entirely on the tkinter thread."""
        root = tk.Tk()
        root.overrideredirect(True)          # no title bar / borders
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.0)       # start fully transparent
        root.configure(bg=self._hex_color())

        # Cover the full primary monitor
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")

        half_ms   = int((self.duration * 1000) / 2)
        step_ms   = max(1, half_ms // self.steps)
        step_size = 1.0 / self.steps

        # State machine: "in" → "swap" → "out" → "done"
        phase     = {"value": "in"}
        alpha     = {"value": 0.0}

        def tick():
            if phase["value"] == "in":
                alpha["value"] = min(1.0, alpha["value"] + step_size)
                root.attributes("-alpha", alpha["value"])
                if alpha["value"] >= 1.0:
                    phase["value"] = "swap"
                    root.after(1, tick)
                else:
                    root.after(step_ms, tick)

            elif phase["value"] == "swap":
                # Screen is fully black — do the window swap now
                try:
                    swap_fn()
                except Exception as exc:
                    print(f"[TRANSITION] swap_fn raised: {exc}")
                phase["value"] = "out"
                root.after(step_ms, tick)

            elif phase["value"] == "out":
                alpha["value"] = max(0.0, alpha["value"] - step_size)
                root.attributes("-alpha", alpha["value"])
                if alpha["value"] <= 0.0:
                    phase["value"] = "done"
                    root.after(1, tick)
                else:
                    root.after(step_ms, tick)

            elif phase["value"] == "done":
                root.destroy()
                done.set()

        root.after(0, tick)
        root.mainloop()
