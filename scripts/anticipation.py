"""
Anticipation Indicator

An always-on-top, borderless overlay that appears as soon as the detector
starts accumulating zone hits, giving the viewer visual feedback that the
system has noticed them — before the full transition fires.

Visual design:
  - A thin arc drawn on a transparent window, anchored to a corner of the
    screen (default: bottom-right).
  - The arc fills clockwise from 0° → 360° as consecutive_hits climbs
    toward min_hits.
  - Colour shifts from a "cold" start colour toward a "hot" ready colour
    as the arc fills.
  - When presence_on fires the overlay dismisses itself instantly.
  - When hits reset to 0 (person left before threshold) the arc drains
    back to empty and the overlay hides.

Thread model:
  - AnticipationOverlay owns a single persistent tkinter thread for its
    entire lifetime (created on first show(), destroyed on close()).
  - The orchestrator calls show(hits, min_hits) / hide() from its own
    thread — all cross-thread communication goes through a queue so
    tkinter stays single-threaded.

Usage:
    overlay = AnticipationOverlay(cfg)
    overlay.show(hits=1, min_hits=4)   # arc at 25%
    overlay.show(hits=3, min_hits=4)   # arc at 75%
    overlay.hide()                     # dismiss (presence fired or reset)
    overlay.close()                    # permanent shutdown
"""

import math
import queue
import threading
import tkinter as tk
from typing import Tuple


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _lerp_color(
    cold: Tuple[int, int, int],
    hot: Tuple[int, int, int],
    t: float,
) -> str:
    """Linearly interpolate between two RGB colours, return as #rrggbb."""
    r = int(cold[0] + (hot[0] - cold[0]) * t)
    g = int(cold[1] + (hot[1] - cold[1]) * t)
    b = int(cold[2] + (hot[2] - cold[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

class AnticipationOverlay:
    """
    Persistent always-on-top arc overlay driven by detection warmup progress.

    Args:
        cfg: The 'anticipation' sub-dict from config.json.  All keys are
             optional — defaults are applied for anything missing.
    """

    # Defaults (overridden by config)
    _DEFAULTS = {
        "enabled":      True,
        "size":         120,          # diameter of the arc widget in pixels
        "thickness":    10,           # arc stroke width in pixels
        "corner":       "br",         # tl / tr / bl / br
        "margin":       30,           # pixels from screen edge
        "color_cold":   [80, 80, 220],   # RGB — arc start colour (few hits)
        "color_hot":    [0, 220, 80],    # RGB — arc end colour (about to fire)
        "bg_opacity":   0.0,          # background fill opacity (0 = fully transparent)
        "drain_steps":  12,           # animation steps when draining back to 0
        "drain_ms":     20,           # ms per drain step
    }

    def __init__(self, cfg: dict):
        merged = {**self._DEFAULTS, **cfg}
        self._enabled     = merged["enabled"]
        self._size        = merged["size"]
        self._thickness   = merged["thickness"]
        self._corner      = merged["corner"]
        self._margin      = merged["margin"]
        self._cold        = tuple(merged["color_cold"])
        self._hot         = tuple(merged["color_hot"])
        self._bg_opacity  = merged["bg_opacity"]
        self._drain_steps = merged["drain_steps"]
        self._drain_ms    = merged["drain_ms"]

        self._cmd_queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Public API  (called from orchestrator thread)
    # ------------------------------------------------------------------

    def show(self, hits: int, min_hits: int) -> None:
        """Update the arc to reflect current warmup progress."""
        if not self._enabled:
            return
        self._ensure_started()
        self._cmd_queue.put(("show", hits, min_hits))

    def hide(self) -> None:
        """Dismiss the overlay (presence fired or hits reset)."""
        if not self._enabled:
            return
        self._cmd_queue.put(("hide",))

    def close(self) -> None:
        """Permanently destroy the overlay window and stop the thread."""
        if not self._enabled or not self._started:
            return
        self._cmd_queue.put(("close",))
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal — tkinter thread
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(
                target=self._tk_main, daemon=True
            )
            self._thread.start()

    def _tk_main(self) -> None:
        """Entire tkinter lifecycle runs here."""
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.0)       # start hidden
        root.configure(bg="black")
        root.attributes("-transparentcolor", "black")  # punch-through bg

        pad    = self._thickness + 4
        total  = self._size + pad * 2
        canvas = tk.Canvas(
            root,
            width=total, height=total,
            bg="black", highlightthickness=0,
        )
        canvas.pack()

        # Position window in chosen corner
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        m  = self._margin
        c  = self._corner
        wx = (sw - total - m) if c in ("tr", "br") else m
        wy = (sh - total - m) if c in ("bl", "br") else m
        root.geometry(f"{total}x{total}+{wx}+{wy}")

        # Arc item — updated each frame
        arc_id = canvas.create_arc(
            pad, pad, pad + self._size, pad + self._size,
            start=90, extent=0,
            style=tk.ARC,
            outline=_lerp_color(self._cold, self._hot, 0.0),
            width=self._thickness,
        )

        # State
        state      = {"visible": False, "progress": 0.0, "draining": False}
        drain_step = {"remaining": 0}

        def _set_arc(progress: float) -> None:
            """Redraw arc for 0.0–1.0 progress."""
            extent = -360.0 * progress          # negative = clockwise
            color  = _lerp_color(self._cold, self._hot, progress)
            canvas.itemconfig(arc_id, extent=extent, outline=color)

        def _drain_tick() -> None:
            """Animate the arc draining back to zero."""
            if drain_step["remaining"] <= 0:
                state["draining"] = False
                state["progress"] = 0.0
                _set_arc(0.0)
                root.attributes("-alpha", 0.0)
                state["visible"] = False
                return
            drain_step["remaining"] -= 1
            t = drain_step["remaining"] / self._drain_steps
            state["progress"] = t
            _set_arc(t)
            root.after(self._drain_ms, _drain_tick)

        def _poll() -> None:
            """Drain the command queue and update the overlay."""
            try:
                while True:
                    cmd = self._cmd_queue.get_nowait()

                    if cmd[0] == "show":
                        _, hits, min_hits = cmd
                        progress = min(1.0, hits / max(min_hits, 1))
                        state["progress"]  = progress
                        state["draining"]  = False
                        drain_step["remaining"] = 0

                        _set_arc(progress)
                        if not state["visible"]:
                            root.attributes("-alpha", 1.0)
                            state["visible"] = True

                    elif cmd[0] == "hide":
                        if state["visible"] and not state["draining"]:
                            state["draining"] = True
                            drain_step["remaining"] = self._drain_steps
                            _drain_tick()

                    elif cmd[0] == "close":
                        root.destroy()
                        return

            except queue.Empty:
                pass

            root.after(30, _poll)

        root.after(0, _poll)
        root.mainloop()
