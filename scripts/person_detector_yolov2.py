#!/usr/bin/env python3
"""
Person Detector v2 using YOLOv8 — with Detection Zone (ROI)

Adds a configurable Detection Zone to person_detector_yolo.py.
Only detections whose **feet** (bottom-centre of bounding box) fall inside
the zone count toward presence. Everything outside is drawn but ignored.

Zone shape
----------
The zone can be defined as either:

  • A trapezoid (recommended for front-facing cameras looking at a floor zone).
    Perspective makes the near edge wider and the far edge narrower, so a
    rectangle cannot model the real zone accurately.

    Pass --zone-poly as a JSON array of four [x, y] normalised points:
        bottom-left → bottom-right → top-right → top-left

    Example:
        --zone-poly "[[0.10,0.90],[0.50,0.90],[0.40,0.62],[0.18,0.62]]"

  • A rectangle (legacy fallback, used when --zone-poly is not supplied).
    Pass --zone-x1/y1/x2/y2 as fractions of frame size (0.0–1.0).

Hit test
--------
Presence is triggered when a person's feet (bottom-centre of their bounding
box) fall inside the polygon, matching the orchestrator's logic exactly.

Visual feedback
---------------
  - Polygon drawn in blue (scanning) or red (presence active)
  - Green dot at the feet position of each detected person
  - Green boxes = feet inside zone (counts toward presence)
  - Grey boxes  = feet outside zone (ignored)
  - Zone border pulses red when presence is active, blue when scanning
  - Press 'r' to reset zone to defaults, 'q' to quit

All v1 robustness features are preserved:
  - Time-based presence persistence
  - Minimum consecutive hit count
  - Frame-grab failure tolerance
  - Detection interval (run inference every N frames)
  - Camera resolution verification
  - FPS overlay
  - try/finally cleanup

Author: Clock-Verbal Team
Date: 2026-05-28

Usage:
    python person_detector_yolov2.py [options]

    --camera-id          Camera index (default: 1)
    --model              YOLOv8 variant: n/s/m/l/x (default: n)
    --confidence         Detection confidence threshold 0-1 (default: 0.45)
    --iou                NMS IOU threshold 0-1 (default: 0.45)
    --detection-interval Run detection every N frames (default: 2)
    --presence-timeout   Seconds of no detection before clearing presence (default: 1.5)
    --min-hits           Consecutive zone-hits required to set presence (default: 2)
    --max-failures       Consecutive frame-grab failures before exit (default: 10)
    --width              Requested capture width (default: 1280)
    --height             Requested capture height (default: 720)
    --zone-poly          Zone as JSON array of 4 [x,y] normalised points (preferred)
    --zone-x1/y1/x2/y2  Legacy rectangular zone (used when --zone-poly is absent)
    --zone-opacity       Tint opacity of the zone fill 0-1 (default: 0.08)
"""

import sys
import json
import time
import argparse
from collections import deque
from typing import List, Tuple, Optional

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

# Default trapezoid — matches config.json zone_poly default
DEFAULT_ZONE_POLY = [
    [0.0, 0.73],
    [0.87, 0.82],
    [0.72, 0.62],
    [0.0, 0.65],
]

# Colours (BGR)
COLOR_BOX_IN_ZONE  = (0, 255, 0)      # green  — person inside zone
COLOR_BOX_OUT_ZONE = (120, 120, 120)  # grey   — person outside zone
COLOR_ZONE_SCAN    = (255, 180, 0)    # blue   — zone border while scanning
COLOR_ZONE_ACTIVE  = (0, 0, 255)      # red    — zone border while presence active
COLOR_ZONE_FILL    = (255, 220, 100)  # light tint for zone fill
COLOR_PRESENT      = (0, 0, 255)      # red    — status banner
COLOR_SCANNING     = (255, 255, 0)    # yellow — status banner
COLOR_FPS          = (200, 200, 200)  # grey   — FPS / hint text
COLOR_LABEL        = (255, 255, 255)  # white  — box confidence label
COLOR_FEET         = (0, 255, 0)      # green  — feet dot


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def build_zone_pts(
    frame_w: int,
    frame_h: int,
    poly_norm: List[List[float]],
) -> np.ndarray:
    """Convert normalised [x, y] polygon points to integer pixel coords."""
    return np.array(
        [[int(p[0] * frame_w), int(p[1] * frame_h)] for p in poly_norm],
        dtype=np.int32,
    )


def feet_in_zone(
    x1: int, y1: int, x2: int, y2: int,
    zone_pts: np.ndarray,
) -> bool:
    """Return True if the bottom-centre of the bbox is inside the polygon."""
    feet_x = (x1 + x2) // 2
    feet_y = y2
    return cv2.pointPolygonTest(zone_pts, (float(feet_x), float(feet_y)), False) >= 0


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
    zone_pts: np.ndarray,
    presence: bool,
    opacity: float,
) -> None:
    """Draw the detection zone: semi-transparent fill + coloured polygon border."""
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], COLOR_ZONE_FILL)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    # Border — colour reflects presence state
    border_color = COLOR_ZONE_ACTIVE if presence else COLOR_ZONE_SCAN
    border_thickness = 3 if presence else 2
    cv2.polylines(frame, [zone_pts], isClosed=True, color=border_color,
                  thickness=border_thickness)

    # Corner tick marks
    tick = 14
    for pt in zone_pts:
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(frame, (cx, cy), 4, border_color, -1)

    # Zone label — anchored to the topmost point
    top_pt = zone_pts[zone_pts[:, 1].argmin()]
    label = "DETECTION ZONE"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx = int(top_pt[0]) + 6
    ly = int(top_pt[1]) - 6
    cv2.putText(
        frame, label, (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA,
    )


def draw_boxes(
    frame: np.ndarray,
    boxes_in: List[Tuple],
    boxes_out: List[Tuple],
) -> None:
    """Draw detection boxes and feet dots, coloured by zone membership."""
    for (x1, y1, x2, y2, conf) in boxes_in:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX_IN_ZONE, 2)
        label = f"{conf:.2f}"
        cv2.putText(
            frame, label, (x1 + 4, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA,
        )
        # Feet dot
        feet_x = (x1 + x2) // 2
        cv2.circle(frame, (feet_x, y2), 5, COLOR_FEET, -1)

    for (x1, y1, x2, y2, conf) in boxes_out:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX_OUT_ZONE, 1)
        feet_x = (x1 + x2) // 2
        cv2.circle(frame, (feet_x, y2), 4, COLOR_BOX_OUT_ZONE, -1)


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

    # --- Resolve zone polygon ---
    # Prefer --zone-poly; fall back to legacy --zone-x1/y1/x2/y2 rect.
    if args.zone_poly is not None:
        zone_norm: List[List[float]] = args.zone_poly
    elif any(v is not None for v in [args.zone_x1, args.zone_y1, args.zone_x2, args.zone_y2]):
        x1 = args.zone_x1 if args.zone_x1 is not None else 0.08
        y1 = args.zone_y1 if args.zone_y1 is not None else 0.62
        x2 = args.zone_x2 if args.zone_x2 is not None else 0.55
        y2 = args.zone_y2 if args.zone_y2 is not None else 0.90
        zone_norm = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        print("INFO: Using legacy rect zone args — converted to polygon.")
    else:
        zone_norm = DEFAULT_ZONE_POLY

    zone_default = [pt[:] for pt in zone_norm]

    # --- Open camera ---
    cap = open_camera(args.camera_id, args.width, args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # State
    presence_active = False
    last_detection_time: float = 0.0
    consecutive_hits: int = 0
    consecutive_failures: int = 0
    frame_index: int = 0

    frame_times: deque = deque(maxlen=30)
    last_frame_time = time.monotonic()

    last_boxes_in: List[Tuple] = []
    last_boxes_out: List[Tuple] = []

    print("Running. Press 'q' to quit, 'r' to reset zone.")
    print(
        f"  model={model_name}  conf={args.confidence}  iou={args.iou}  "
        f"interval={args.detection_interval}  presence_timeout={args.presence_timeout}s  "
        f"min_hits={args.min_hits}"
    )
    print(f"  zone_poly={zone_norm}  opacity={args.zone_opacity}")

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
            zone_pts = build_zone_pts(fw, fh, zone_norm)

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
                        if feet_in_zone(x1, y1, x2, y2, zone_pts):
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
            draw_zone(frame, zone_pts, presence_active, args.zone_opacity)
            draw_boxes(frame, last_boxes_in, last_boxes_out)
            draw_status(frame, presence_active, fps, consecutive_hits)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("r"):
                zone_norm = [pt[:] for pt in zone_default]
                print(f"Zone reset to default: {zone_norm}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_zone_poly(value: str) -> List[List[float]]:
    """Parse --zone-poly JSON string into a list of [x, y] pairs."""
    try:
        pts = json.loads(value)
        if (
            not isinstance(pts, list)
            or len(pts) < 3
            or not all(isinstance(p, list) and len(p) == 2 for p in pts)
        ):
            raise ValueError
        return [[float(p[0]), float(p[1])] for p in pts]
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "zone-poly must be a JSON array of [x,y] pairs, e.g. "
            '"[[0.10,0.90],[0.50,0.90],[0.40,0.62],[0.18,0.62]]"'
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person detection with trapezoidal Detection Zone using YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Camera / model
    p.add_argument("--camera-id", "-c", type=int, default=1,
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
    # Zone — trapezoid (preferred)
    p.add_argument(
        "--zone-poly", type=parse_zone_poly, default=None,
        metavar="JSON",
        help=(
            'Trapezoidal zone as JSON array of 4 [x,y] normalised points '
            '(bottom-left, bottom-right, top-right, top-left). '
            'Example: "[[0.10,0.90],[0.50,0.90],[0.40,0.62],[0.18,0.62]]"'
        ),
    )
    # Zone — legacy rectangle (fallback when --zone-poly is absent)
    p.add_argument("--zone-x1", type=float, default=None,
                   help="[Legacy] Zone left edge as fraction of frame width")
    p.add_argument("--zone-y1", type=float, default=None,
                   help="[Legacy] Zone top edge as fraction of frame height")
    p.add_argument("--zone-x2", type=float, default=None,
                   help="[Legacy] Zone right edge as fraction of frame width")
    p.add_argument("--zone-y2", type=float, default=None,
                   help="[Legacy] Zone bottom edge as fraction of frame height")
    p.add_argument("--zone-opacity", type=float, default=0.08,
                   help="Opacity of the zone fill tint (0–1)")
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
