#!/usr/bin/env python3
"""
Person Detector using YOLOv8 (RGB Fallback)

Robust replacement for the HOG-based person_detector.py.
Uses Ultralytics YOLOv8 for significantly better accuracy and lower
false-positive rates. Falls back gracefully on camera errors.

Key improvements over HOG version:
  - YOLOv8 model with configurable confidence + IOU thresholds
  - Time-based presence persistence (not frame-count-based)
  - Minimum consecutive hit count before declaring presence
  - Consecutive frame-grab failure tolerance before aborting
  - Detection runs on a configurable interval; display runs every frame
  - Camera resolution verified after setting
  - FPS overlay for live performance monitoring
  - All tuning parameters exposed as CLI args
  - try/finally guarantees camera + window cleanup

Author: Clock-Verbal Team
Date: 2026-05-22

Usage:
    python person_detector_yolo.py [options]

    --camera-id          Camera index (default: 0)
    --model              YOLOv8 model variant: n/s/m/l/x (default: n)
    --confidence         Detection confidence threshold 0-1 (default: 0.45)
    --iou                NMS IOU threshold 0-1 (default: 0.45)
    --detection-interval Run detection every N frames (default: 2)
    --presence-timeout   Seconds of no detection before clearing presence (default: 1.5)
    --min-hits           Consecutive detections required to set presence (default: 2)
    --max-failures       Consecutive frame-grab failures before exit (default: 10)
    --width              Requested capture width (default: 1280)
    --height             Requested capture height (default: 720)
"""

import sys
import time
import argparse
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional YOLO import — give a clear install hint if missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: 'ultralytics' package not found.\n"
        "Install it with:  pip install ultralytics\n"
        "The package also requires opencv-python and numpy (already in requirements.txt)."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_PERSON_CLASS_ID = 0          # YOLOv8 COCO class index for 'person'
WINDOW_NAME = "Person Detector (YOLOv8)"
COLOR_BOX = (0, 255, 0)           # green — detection bounding box
COLOR_PRESENT = (0, 0, 255)       # red   — presence active
COLOR_SCANNING = (255, 255, 0)    # yellow — scanning
COLOR_FPS = (200, 200, 200)       # light grey — FPS text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera and request resolution; warn if not honoured."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != width or actual_h != height:
        print(
            f"WARNING: Requested {width}x{height} but camera returned {actual_w}x{actual_h}. "
            "Continuing with actual resolution."
        )
    else:
        print(f"Camera {camera_id} opened at {actual_w}x{actual_h}.")

    return cap


def draw_overlay(
    frame: np.ndarray,
    boxes,
    presence: bool,
    fps: float,
) -> None:
    """Draw bounding boxes, status banner, and FPS counter onto frame in-place."""
    h, w = frame.shape[:2]

    # Bounding boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    # Status banner
    if presence:
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_PRESENT, 5)
        cv2.putText(
            frame, "Presence Detected!", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_PRESENT, 3, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "Scanning...", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SCANNING, 2, cv2.LINE_AA,
        )

    # FPS counter (bottom-left)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FPS, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------

def run_detector(args: argparse.Namespace) -> None:
    """Main detection loop."""

    # --- Load model ---
    model_name = f"yolov8{args.model}.pt"
    print(f"Loading {model_name} (downloads automatically on first run)...")
    model = YOLO(model_name)
    print("Model ready.")

    # --- Open camera ---
    cap = open_camera(args.camera_id, args.width, args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # State
    presence_active = False
    last_detection_time: float = 0.0   # wall-clock time of most recent detection
    consecutive_hits: int = 0          # detections in a row (for min-hits gate)
    consecutive_failures: int = 0      # consecutive frame-grab failures
    frame_index: int = 0

    # FPS tracking — rolling window of last 30 frame durations
    frame_times: deque = deque(maxlen=30)
    last_frame_time = time.monotonic()

    # Cache last detection boxes so they stay visible between detection frames
    last_boxes = []

    print(f"Running. Press 'q' to quit.")
    print(
        f"  model={model_name}  conf={args.confidence}  iou={args.iou}  "
        f"interval={args.detection_interval}  presence_timeout={args.presence_timeout}s  "
        f"min_hits={args.min_hits}"
    )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                print(
                    f"WARNING: Frame grab failed "
                    f"({consecutive_failures}/{args.max_failures})"
                )
                if consecutive_failures >= args.max_failures:
                    print("ERROR: Too many consecutive frame failures. Exiting.")
                    break
                time.sleep(0.05)
                continue

            # Successful grab — reset failure counter
            consecutive_failures = 0

            # --- FPS calculation ---
            now = time.monotonic()
            frame_times.append(now - last_frame_time)
            last_frame_time = now
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            # --- Run detection on every Nth frame ---
            if frame_index % args.detection_interval == 0:
                results = model(
                    frame,
                    classes=[COCO_PERSON_CLASS_ID],
                    conf=args.confidence,
                    iou=args.iou,
                    verbose=False,
                )

                # Extract integer bounding boxes for detected persons
                last_boxes = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        last_boxes.append((x1, y1, x2, y2))

                # --- Presence state machine ---
                if last_boxes:
                    consecutive_hits += 1
                    last_detection_time = now
                    if consecutive_hits >= args.min_hits:
                        presence_active = True
                else:
                    consecutive_hits = 0
                    # Keep presence active until timeout expires
                    if presence_active and (now - last_detection_time) > args.presence_timeout:
                        presence_active = False

            frame_index += 1

            # --- Draw and display ---
            draw_overlay(frame, last_boxes, presence_active, fps)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit requested.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust person detection using YOLOv8 (RGB camera)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera-id", "-c", type=int, default=0,
                   help="Camera device index")
    p.add_argument("--model", "-m", choices=["n", "s", "m", "l", "x"], default="n",
                   help="YOLOv8 model size (n=nano … x=extra-large)")
    p.add_argument("--confidence", type=float, default=0.45,
                   help="Minimum detection confidence (0–1)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IOU threshold (0–1)")
    p.add_argument("--detection-interval", type=int, default=2,
                   help="Run detection every N frames (1 = every frame)")
    p.add_argument("--presence-timeout", type=float, default=1.5,
                   help="Seconds without detection before clearing presence flag")
    p.add_argument("--min-hits", type=int, default=2,
                   help="Consecutive detections required before presence is declared")
    p.add_argument("--max-failures", type=int, default=10,
                   help="Consecutive frame-grab failures before aborting")
    p.add_argument("--width", type=int, default=1280,
                   help="Requested capture width in pixels")
    p.add_argument("--height", type=int, default=720,
                   help="Requested capture height in pixels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_detector(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
