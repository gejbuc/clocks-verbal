"""
Transition — Fade-to-Black Overlay

Creates a borderless, always-on-top fullscreen window that fades to black
and back, masking the hard window swap underneath.

The overlay runs on its own thread so the caller just does:

    fade = FadeTransition(duration=0.4, color=(0, 0, 0))
    fade.run(swap_fn)          # blocks until the full fade-in/swap/fade-out is done

`swap_fn` is called at peak opacity (screen fully black) — that's the moment
the underlying windows are rearranged, completely hidden from the viewer.

Thread model:
  - A single persistent tkinter thread is created on the first call to run().
  - Subsequent calls reuse the same thread and Tk() root, avoiding the
    Tcl_AsyncDelete crash that occurs when multiple Tk() instances are
    created on different threads.

Dependencies: tkinter (stdlib), no extra installs needed.
"""

import threading
import queue
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

        self._cmd_queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, swap_fn: Callable[[], None]) -> None:
        """
        Block until the full transition completes:
          1. Fade in (transparent → opaque)
          2. Call swap_fn() at peak opacity
          3. Fade out (opaque → transparent)
          4. Hide the overlay window (kept alive for reuse)

        Args:
            swap_fn: Zero-argument callable executed at peak opacity.
                     This is where you minimize/focus windows.
        """
        self._ensure_started()
        done = threading.Event()
        self._cmd_queue.put(("fade", swap_fn, done))
        done.wait()

    def close(self) -> None:
        """Permanently destroy the overlay window and stop the thread."""
        if not self._started:
            return
        self._cmd_queue.put(("close",))
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hex_color(self) -> str:
        r, g, b = self.color
        return f"#{r:02x}{g:02x}{b:02x}"

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(
                target=self._tk_main, daemon=True
            )
            self._thread.start()

    def _tk_main(self) -> None:
        """Entire tkinter lifecycle runs here — single thread, single Tk root."""
        root = tk.Tk()
        root.overrideredirect(True)          # no title bar / borders
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.0)       # start fully transparent
        root.configure(bg=self._hex_color())
        root.withdraw()                      # start hidden

        # Cover the full primary monitor
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")

        def _poll() -> None:
            """Drain the command queue."""
            try:
                while True:
                    cmd = self._cmd_queue.get_nowait()

                    if cmd[0] == "fade":
                        _, swap_fn, done_event = cmd
                        _start_fade(swap_fn, done_event)
                        return  # _start_fade re-registers _poll when done

                    elif cmd[0] == "close":
                        root.destroy()
                        return

            except queue.Empty:
                pass

            root.after(30, _poll)

        def _start_fade(
            swap_fn: Callable[[], None],
            done_event: threading.Event,
        ) -> None:
            """Kick off a fade-in / swap / fade-out cycle."""
            half_ms   = int((self.duration * 1000) / 2)
            step_ms   = max(1, half_ms // self.steps)
            step_size = 1.0 / self.steps

            # Show the overlay window
            root.deiconify()
            root.attributes("-alpha", 0.0)

            phase = {"value": "in"}
            alpha = {"value": 0.0}

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
                    root.withdraw()       # hide but keep alive for reuse
                    done_event.set()
                    root.after(0, _poll)   # resume polling for next command

            root.after(0, tick)

        root.after(0, _poll)
        root.mainloop()
