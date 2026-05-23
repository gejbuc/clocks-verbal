"""
Window Manager

Handles focusing and minimizing windows by title substring, and launching
background processes (used for the idle video player).

Dependencies:
    pip install pygetwindow
"""

import subprocess
import time
import ctypes
from typing import Optional

try:
    import pygetwindow as gw
except ImportError:
    print("[ERROR] pygetwindow not installed. Run: pip install pygetwindow")
    raise


class WindowManager:
    """Title-based window focus/minimize and process launching."""

    # ------------------------------------------------------------------
    # Process launching
    # ------------------------------------------------------------------

    def launch(self, cmd: list[str]) -> Optional[subprocess.Popen]:
        """
        Launch a command as a non-blocking background process.

        Args:
            cmd: Command as a list, e.g. ["vlc", "C:/media/idle.mp4", "--loop"]

        Returns:
            Popen handle on success, None on failure.
        """
        try:
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            print(f"[INFO] Launched PID {process.pid}: {' '.join(cmd)}")
            return process
        except FileNotFoundError:
            print(f"[ERROR] Executable not found: {cmd[0]}")
            return None
        except Exception as exc:
            print(f"[ERROR] Failed to launch {cmd[0]}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Window focus
    # ------------------------------------------------------------------

    def focus(
        self,
        title: str,
        exclude: Optional[str] = None,
        maximize: bool = True,
        delay: float = 0.0,
    ) -> bool:
        """
        Bring a window to the foreground by title substring.

        Uses the Windows ALT-key bypass to work around focus-stealing
        protection that would otherwise silently fail.

        Args:
            title:    Substring to match against window titles.
            exclude:  Optional substring — windows containing this are skipped.
            maximize: Maximize the window after focusing.
            delay:    Optional sleep (seconds) after each window operation,
                      useful if the app needs time to respond.

        Returns:
            True if a matching window was successfully focused, False otherwise.
        """
        try:
            matches = gw.getWindowsWithTitle(title)
            if exclude:
                matches = [w for w in matches if exclude.lower() not in w.title.lower()]

            if not matches:
                print(f"[WARN] No window found matching '{title}'")
                return False

            win = matches[0]

            # Already in front and not minimized — just maximize if needed
            if win.isActive and not win.isMinimized:
                if maximize and not win.isMaximized:
                    try:
                        win.maximize()
                    except Exception:
                        pass
                return True

            # ALT-key bypass: tricks Windows into allowing the focus change
            ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)   # ALT down
            ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)   # ALT up

            if win.isMinimized:
                win.restore()
                if delay > 0:
                    time.sleep(delay)

            if maximize and not win.isMaximized:
                try:
                    win.maximize()
                    if delay > 0:
                        time.sleep(delay)
                except Exception:
                    pass

            try:
                win.activate()
            except Exception as exc:
                print(f"[WARN] activate() failed for '{win.title}': {exc}")

            if delay > 0:
                time.sleep(delay)

            return win.isActive

        except Exception as exc:
            print(f"[ERROR] Unexpected error focusing '{title}': {exc}")
            return False

    # ------------------------------------------------------------------
    # Window minimize
    # ------------------------------------------------------------------

    def minimize(self, title: str, exclude: Optional[str] = None) -> bool:
        """
        Minimize a window by title substring.

        Args:
            title:   Substring to match against window titles.
            exclude: Optional substring — windows containing this are skipped.

        Returns:
            True if a matching window was found and minimized, False otherwise.
        """
        try:
            matches = gw.getWindowsWithTitle(title)
            if exclude:
                matches = [w for w in matches if exclude.lower() not in w.title.lower()]

            if not matches:
                return False

            win = matches[0]
            if not win.isMinimized:
                win.minimize()
            return True

        except Exception as exc:
            print(f"[ERROR] Failed to minimize '{title}': {exc}")
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_running(self, title: str, exclude: Optional[str] = None) -> bool:
        """Return True if any window matching the title substring exists."""
        matches = gw.getWindowsWithTitle(title)
        if exclude:
            matches = [w for w in matches if exclude.lower() not in w.title.lower()]
        return len(matches) > 0
