#!/usr/bin/env python3
"""
Orchestrator — Clocks Verbal

Ties together the YOLOv8 person detector and the window manager into a
two-state kiosk loop:

  IDLE   — idle video plays in the foreground
  ACTIVE — tracking app window is brought to the foreground

State transitions:
  IDLE   → ACTIVE  when a person is detected inside the detection zone
                   for at least `min_hits` consecutive detection frames
  ACTIVE → IDLE    when no person has been in the zone for
                   `grace_period_seconds`

The tracking exe is assumed to be already running independently.
If its window cannot be found when presence is detected, a warning is
logged and the system stays on IDLE until the window appears.

The idle video player is launched by the orchestrator on startup and
kept alive for the duration of the session.

Usage:
    python orchestrator.py [--config path/to/config.json]

Config file: see config.json in the project root.
"""

import sys
import time
import queue
import threading
import argparse
import json
import subprocess
from pathlib import Path
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

from window_manager import WindowManager


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE   = auto()
    ACTIVE = auto()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "camera_id": 1,
    "detection": {
        "model": "n",
        "confidence": 0.45,
        "iou": 0.45,
        "detection_interval": 2,
        "presence_timeout": 2.0,
        "min_hits": 2,
        "max_failures": 10,
        "zone": {"x1": 0.08, "y1": 0.62, "x2": 0.55, "y2": 0.90},
    },
    "active_window_title": "",
    "idle_video_path": "",
    "idle_player_title": "VLC media player",
    "grace_period_seconds": 3.0,
    "switch_delay_seconds": 0.3,
}


def load_config(path: str) -> dict:
    """Load config.json, merging with defaults for any missing keys."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy of defaults
    try:
        with open(path, "r") as f:
            user = json.load(f)
        # Top-level merge
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        print(f"[INFO] Config loaded from {path}")
    except FileNotFoundError:
        print(f"[WARN] Config file not found at '{path}'. Using defaults.")
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in config: {exc}. Using defaults.")
    return cfg


# ---------------------------------------------------------------------------
# Idle video player
# ---------------------------------------------------------------------------

class IdlePlayer:
    """Manages the idle video player process lifecycle."""

    def __init__(self, video_path: str, player_title: str, wm: WindowManager):
        self._path   = video_path
        self._title  = player_title
        self._wm     = wm
        self._proc: Optional[subprocess.Popen] = None

    def ensure_running(self) -> bool:
        """Launch the player if it isn't already visible. Returns True if running."""
        if not self._path:
            return False  # no idle video configured

        if self._wm.is_running(self._title):
            return True

        # (Re)launch VLC with loop flag — adjust if using a different player
        cmd = ["vlc", "--loop", "--fullscreen", self._path]
        self._proc = self._wm.launch(cmd)
        if self._proc:
            time.sleep(1.5)  # give the player a moment to open
        return self._proc is not None

    def bring_forward(self, delay: float = 0.0) -> None:
        self.ensure_running()
        self._wm.focus(self._title, delay=delay)

    def send_back(self) -> None:
        self._wm.minimize(self._title)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None


# ---------------------------------------------------------------------------
# Detector thread
# ---------------------------------------------------------------------------

COCO_PERSON = 0


def detector_thread(cfg: dict, event_queue: queue.Queue, stop_event: threading.Event) -> None:
    """
    Runs the YOLOv8 detection loop in a background thread.

    Puts events onto event_queue:
        "presence_on"  — person entered zone (min_hits threshold met)
        "presence_off" — person left zone (presence_timeout expired)
    """
    det = cfg["detection"]
    cam_id   = cfg["camera_id"]
    model_id = f"yolov8{det['model']}.pt"
    conf     = det["confidence"]
    iou      = det["iou"]
    interval = det["detection_interval"]
    timeout  = det["presence_timeout"]
    min_hits = det["min_hits"]
    max_fail = det["max_failures"]
    zone     = det["zone"]

    print(f"[DETECTOR] Loading {model_id}...")
    model = YOLO(model_id)
    print(f"[DETECTOR] Model ready.")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[DETECTOR] ERROR: Cannot open camera {cam_id}")
        event_queue.put("error")
        return

    presence_active   = False
    last_detect_time  = 0.0
    consecutive_hits  = 0
    consecutive_fails = 0
    frame_idx         = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                consecutive_fails += 1
                if consecutive_fails >= max_fail:
                    print("[DETECTOR] ERROR: Too many frame failures. Stopping.")
                    event_queue.put("error")
                    break
                time.sleep(0.05)
                continue

            consecutive_fails = 0
            now = time.monotonic()

            if frame_idx % interval == 0:
                fh, fw = frame.shape[:2]
                zx1 = int(zone["x1"] * fw)
                zy1 = int(zone["y1"] * fh)
                zx2 = int(zone["x2"] * fw)
                zy2 = int(zone["y2"] * fh)

                results = model(
                    frame,
                    classes=[COCO_PERSON],
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )

                zone_hit = False
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        if x2 > zx1 and x1 < zx2 and y2 > zy1 and y1 < zy2:
                            zone_hit = True
                            break

                if zone_hit:
                    consecutive_hits += 1
                    last_detect_time = now
                    if not presence_active and consecutive_hits >= min_hits:
                        presence_active = True
                        event_queue.put("presence_on")
                        print("[DETECTOR] presence_on")
                else:
                    consecutive_hits = 0
                    if presence_active and (now - last_detect_time) > timeout:
                        presence_active = False
                        event_queue.put("presence_off")
                        print("[DETECTOR] presence_off")

            frame_idx += 1

    finally:
        cap.release()
        print("[DETECTOR] Camera released.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    wm     = WindowManager()
    player = IdlePlayer(cfg["idle_video_path"], cfg["idle_player_title"], wm)
    delay  = cfg["switch_delay_seconds"]

    active_title  = cfg["active_window_title"]
    grace_period  = cfg["grace_period_seconds"]

    state = State.IDLE
    last_active_time = 0.0

    # Start idle video
    print("[ORCH] Starting idle video player...")
    player.bring_forward(delay=delay)

    # Start detector thread
    event_queue  = queue.Queue()
    stop_event   = threading.Event()
    det_thread   = threading.Thread(
        target=detector_thread,
        args=(cfg, event_queue, stop_event),
        daemon=True,
    )
    det_thread.start()
    print("[ORCH] Detector running. Press Ctrl+C to stop.")

    try:
        while True:
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                # Periodic check: if ACTIVE but tracking window disappeared, fall back to IDLE
                if state == State.ACTIVE and not wm.is_running(active_title):
                    print("[ORCH] Tracking window gone. Returning to IDLE.")
                    state = State.IDLE
                    player.bring_forward(delay=delay)
                continue

            if event == "error":
                print("[ORCH] Detector reported an error. Shutting down.")
                break

            elif event == "presence_on":
                if state == State.IDLE:
                    if wm.is_running(active_title):
                        print("[ORCH] Presence detected — switching to ACTIVE.")
                        player.send_back()
                        time.sleep(delay)
                        wm.focus(active_title, delay=delay)
                        state = State.ACTIVE
                        last_active_time = time.monotonic()
                    else:
                        print(
                            f"[WARN] Presence detected but tracking window "
                            f"'{active_title}' not found. Staying IDLE."
                        )

            elif event == "presence_off":
                if state == State.ACTIVE:
                    # Honour grace period before switching back
                    elapsed = time.monotonic() - last_active_time
                    remaining = grace_period - elapsed
                    if remaining > 0:
                        print(f"[ORCH] Presence lost — waiting {remaining:.1f}s grace period.")
                        time.sleep(remaining)
                    print("[ORCH] Grace period done — returning to IDLE.")
                    player.bring_forward(delay=delay)
                    state = State.IDLE

    except KeyboardInterrupt:
        print("\n[ORCH] Interrupted by user.")
    finally:
        stop_event.set()
        det_thread.join(timeout=3.0)
        player.stop()
        print("[ORCH] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clocks Verbal orchestrator — presence-driven window switcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        default=str(Path(__file__).parent.parent / "config.json"),
        help="Path to config.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
