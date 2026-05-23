#!/usr/bin/env python3
"""
Person Detector v2 using YOLOv8 — with Detection Zone (ROI)

Adds a configurable Detection Zone rectangle to person_detector_yolo.py.
Only detections whose bounding box overlaps the zone count toward presence.
Everything outside the zone is drawn but ignored for presence logic.

New in v2:
  - Detection Zone: a semi-transparent rectangle drawn over the live feed.
    Only persons overlapping this zone trigger presence.
  - Zone is defined as fractions of frame size (0.0–1.0) so it works at
    any resolution without hardcoding pixel values.
  - Boxes inside the zone are drawn green; boxes outside are drawn grey.
  - Zone border pulses red when presence is active, blue when scanning.
  - --zone-x1/y1/x2/y2  CLI args to position the zone (default: centre 60%)
  - Press 'r' at runtime to reset the zone to default.

All v1 robustness features are preserved:
  - Time-based presence persistence
  - Minimum consecutive hit count
  - Frame-grab failure tolerance
  - Detection interval (run inference every N frames)
  - Camera resolution verification
  - FPS overlay
  - try/finally cleanup

Author: Clock-Verbal Team
Date: 2026-05-23

Usage:
    python person_detector_yolov2.py [options]

    --camera-id          Camera index (default: 0)
    --model              YOLOv8 variant: n/s/m/l/x (default: n)
    --confidence         Detection confidence threshold 0-1 (default: 0.45)
    --iou                NMS IOU threshold 0-1 (default: 0.45)
    --detection-interval Run detection every N frames (default: 2)
    --presence-timeout   Seconds of no detection before clearing presence (default: 1.5)
    --min-hits           Consecutive zone-hits required to set presence (default: 2)
    --max-failures       Consecutive frame-grab failures before exit (default: 10)
    --width              Requested capture width (default: 1280)
    --height             Requested capture height (default: 720)
    --zone-x1            Zone left edge as fraction of frame width (default: 0.20)
    --zone-y1            Zone top edge as fraction of frame height (default: 0.15)
    --zone-x2            Zone right edge as fraction of frame width (default: 0.80)
    --zone-y2            Zone bottom edge as fraction of frame height (default: 0.85)
    --zone-opacity       Tint opacity of the zone fill 0-1 (default: 0.08)
"""

import sys
import time
import argparse
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional YOLO import
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
COCO_PERSON_CLASS_ID = 0

WINDOW_NAME = "Person Detector v2 (YOLOv8 + Detection Zone)"

# Colours (BGR)
COLOR_BOX_IN_ZONE  = (0, 255, 0)      # green  — person inside zone
COLOR_BOX_OUT_ZONE = (120, 120, 120)  # grey   — person outside zone
COLOR_ZONE_SCAN    = (255, 180, 0)    # blue   — zone border while scanning
COLOR_ZONE_ACTIVE  = (0, 0, 255)      # red    — zone border while presence active
COLOR_ZONE_FILL    = (255, 220, 100)  # light blue tint for zone fill
COLOR_PRESENT      = (0, 0, 255)      # red    — status banner
COLOR_SCANNING     = (255, 255, 0)    # yellow — status banner
COLOR_FPS          = (200, 200, 200)  # grey   — FPS text
COLOR_LABEL        = (255, 255, 255)  # white  — box confidence label


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def zone_pixels(
    frame_w: int, frame_h: int,
    zx1: float, zy1: float, zx2: float, zy2: float,
) -> Tuple[int, int, int, int]:
    """Convert fractional zone coords to pixel coords."""
    return (
        int(zx1 * frame_w),
        int(zy1 * frame_h),
        int(zx2 * frame_w),
        int(zy2 * frame_h),
    )


def box_overlaps_zone(
    bx1: int, by1: int, bx2: int, by2: int,
    zx1: int, zy1: int, zx2: int, zy2: int,
) -> bool:
    """Return True if the detection box overlaps the zone rectangle at all."""
    return bx2 > zx1 and bx1 < zx2 and by2 > zy1 and by1 < zy2


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != width or actual_h != height:
        print(
            f"WARNING: Requested {width}x{height} but camera returned "
            f"{actual_w}x{actual_h}. Continuing with actual resolution."
        )
    else:
        print(f"Camera {camera_id} opened at {actual_w}x{actual_h}.")

    return cap


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_zone(
    frame: np.ndarray,
    zx1: int, zy1: int, zx2: int, zy2: int,
    presence: bool,
    opacity: float,
) -> None:
    """Draw the detection zone: semi-transparent fill + coloured border."""
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), COLOR_ZONE_FILL, -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    # Border — pulses colour based on presence state
    border_color = COLOR_ZONE_ACTIVE if presence else COLOR_ZONE_SCAN
    border_thickness = 3 if presence else 2
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), border_color, border_thickness)

    # Corner tick marks for a cleaner look
    tick = 14
    for (cx, cy), (dx, dy) in [
        ((zx1, zy1), (1, 1)),
        ((zx2, zy1), (-1, 1)),
        ((zx1, zy2), (1, -1)),
        ((zx2, zy2), (-1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy), border_color, 2)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tick), border_color, 2)

    # Zone label
    label = "DETECTION ZONE"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx = zx1 + 6
    ly = zy1 + lh + 6
    cv2.putText(
        frame, label, (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA,
    )


def draw_boxes(
    frame: np.ndarray,
    boxes_in: List[Tuple],
    boxes_out: List[Tuple],
) -> None:
    """Draw detection boxes, coloured by whether they are inside the zone."""
    for (x1, y1, x2, y2, conf) in boxes_in:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX_IN_ZONE, 2)
        label = f"{conf:.2f}"
        cv2.putText(
            frame, label, (x1 + 4, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA,
        )
    for (x1, y1, x2, y2, conf) in boxes_out:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX_OUT_ZONE, 1)


def draw_status(
    frame: np.ndarray,
    presence: bool,
    fps: float,
    zone_hits: int,
) -> None:
    """Draw status banner, FPS, and zone-hit counter."""
    h, w = frame.shape[:2]

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

    # FPS — bottom left
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FPS, 1, cv2.LINE_AA,
    )

    # Zone hit streak — bottom right
    streak_text = f"Zone hits: {zone_hits}"
    (tw, _), _ = cv2.getTextSize(streak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        frame, streak_text, (w - tw - 10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FPS, 1, cv2.LINE_AA,
    )

    # Keyboard hint — top right
    hint = "Q: quit  R: reset zone"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(
        frame, hint, (w - hw - 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_FPS, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------

def run_detector(args: argparse.Namespace) -> None:
    # --- Load model ---
    model_name = f"yolov8{args.model}.pt"
    print(f"Loading {model_name} (downloads automatically on first run)...")
    model = YOLO(model_name)
    print("Model ready.")

    # --- Open camera ---
    cap = open_camera(args.camera_id, args.width, args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Mutable zone fractions (reset with 'r')
    zone = [args.zone_x1, args.zone_y1, args.zone_x2, args.zone_y2]
    zone_default = zone[:]

    # State
    presence_active = False
    last_detection_time: float = 0.0
    consecutive_hits: int = 0
    consecutive_failures: int = 0
    frame_index: int = 0

    frame_times: deque = deque(maxlen=30)
    last_frame_time = time.monotonic()

    # Cached results (kept visible between detection frames)
    last_boxes_in: List[Tuple] = []
    last_boxes_out: List[Tuple] = []

    print("Running. Press 'q' to quit, 'r' to reset zone.")
    print(
        f"  model={model_name}  conf={args.confidence}  iou={args.iou}  "
        f"interval={args.detection_interval}  presence_timeout={args.presence_timeout}s  "
        f"min_hits={args.min_hits}"
    )
    print(
        f"  zone=({zone[0]:.2f},{zone[1]:.2f})→({zone[2]:.2f},{zone[3]:.2f})  "
        f"opacity={args.zone_opacity}"
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

            consecutive_failures = 0

            # FPS
            now = time.monotonic()
            frame_times.append(now - last_frame_time)
            last_frame_time = now
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            fh, fw = frame.shape[:2]
            zx1, zy1, zx2, zy2 = zone_pixels(fw, fh, *zone)

            # --- Detection ---
            if frame_index % args.detection_interval == 0:
                results = model(
                    frame,
                    classes=[COCO_PERSON_CLASS_ID],
                    conf=args.confidence,
                    iou=args.iou,
                    verbose=False,
                )

                last_boxes_in = []
                last_boxes_out = []

                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        if box_overlaps_zone(x1, y1, x2, y2, zx1, zy1, zx2, zy2):
                            last_boxes_in.append((x1, y1, x2, y2, conf))
                        else:
                            last_boxes_out.append((x1, y1, x2, y2, conf))

                # Presence state machine — only zone hits count
                if last_boxes_in:
                    consecutive_hits += 1
                    last_detection_time = now
                    if consecutive_hits >= args.min_hits:
                        presence_active = True
                else:
                    consecutive_hits = 0
                    if presence_active and (now - last_detection_time) > args.presence_timeout:
                        presence_active = False

            frame_index += 1

            # --- Draw ---
            draw_zone(frame, zx1, zy1, zx2, zy2, presence_active, args.zone_opacity)
            draw_boxes(frame, last_boxes_in, last_boxes_out)
            draw_status(frame, presence_active, fps, consecutive_hits)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("r"):
                zone[:] = zone_default
                print(f"Zone reset to default: {zone_default}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person detection with Detection Zone using YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Camera / model
    p.add_argument("--camera-id", "-c", type=int, default=0,
                   help="Camera device index")
    p.add_argument("--model", "-m", choices=["n", "s", "m", "l", "x"], default="n",
                   help="YOLOv8 model size (n=nano … x=extra-large)")
    p.add_argument("--width", type=int, default=1280,
                   help="Requested capture width in pixels")
    p.add_argument("--height", type=int, default=720,
                   help="Requested capture height in pixels")
    # Detection tuning
    p.add_argument("--confidence", type=float, default=0.45,
                   help="Minimum detection confidence (0–1)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IOU threshold (0–1)")
    p.add_argument("--detection-interval", type=int, default=2,
                   help="Run detection every N frames")
    # Presence logic
    p.add_argument("--presence-timeout", type=float, default=1.5,
                   help="Seconds without a zone-hit before clearing presence")
    p.add_argument("--min-hits", type=int, default=2,
                   help="Consecutive zone-hits required before presence is declared")
    p.add_argument("--max-failures", type=int, default=10,
                   help="Consecutive frame-grab failures before aborting")
    # Detection zone (fractional, 0.0–1.0)
    p.add_argument("--zone-x1", type=float, default=0.20,
                   help="Zone left edge as fraction of frame width")
    p.add_argument("--zone-y1", type=float, default=0.15,
                   help="Zone top edge as fraction of frame height")
    p.add_argument("--zone-x2", type=float, default=0.80,
                   help="Zone right edge as fraction of frame width")
    p.add_argument("--zone-y2", type=float, default=0.85,
                   help="Zone bottom edge as fraction of frame height")
    p.add_argument("--zone-opacity", type=float, default=0.08,
                   help="Opacity of the zone fill tint (0–1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Basic zone sanity check
    if not (0.0 <= args.zone_x1 < args.zone_x2 <= 1.0 and
            0.0 <= args.zone_y1 < args.zone_y2 <= 1.0):
        print("ERROR: Zone coordinates must satisfy 0 ≤ x1 < x2 ≤ 1 and 0 ≤ y1 < y2 ≤ 1.")
        sys.exit(1)
    try:
        run_detector(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
