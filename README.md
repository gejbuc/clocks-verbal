# Clocks Verbal

Presence-driven window switcher for kiosk and interactive display setups.

A YOLOv8-based person detector watches a camera feed. When someone steps
into a configurable detection zone, the system brings a target application
(e.g. a tracking or interactive experience) to the foreground. When they
leave, it returns to a looping idle video.

## How it works

```
Camera feed
    │
    ▼
YOLOv8 detector  ──── detection zone (ROI) ────►  warming / presence_on / presence_off
                                                          │
                                          ┌───────────────┼───────────────┐
                                    arc fills up     fade-to-black    arc drains
                                  (anticipation)      (transition)   (reset/idle)
                                          │               │
                                   ┌──────┴──────┐        ▼
                                IDLE state    ACTIVE state
                              (idle video)  (tracking app)
```

**Full experience when someone walks up:**
1. Idle video plays fullscreen
2. Person enters the detection zone → a progress arc appears in the corner, filling blue → green
3. Arc reaches 100% (`min_hits` threshold met) → arc dismisses → fade-to-black → tracking app snaps forward → fade back in
4. Person leaves → grace period → fade-to-black → idle video returns
5. Person leaves before threshold → arc drains back to zero and disappears

**Key design decisions:**
- The tracking app is launched independently — the orchestrator finds it by window title
- The idle video is owned and kept alive by the orchestrator
- If the tracking window is not found when presence fires, a warning is logged and the system stays on idle

## Project Structure

```
clocks-verbal/
├── config.json                    # Runtime configuration (see below)
├── scripts/
│   ├── orchestrator.py            # Main entry point
│   ├── anticipation.py            # Arc overlay — warmup progress indicator
│   ├── transition.py              # Fade-to-black overlay — masks window swaps
│   ├── window_manager.py          # Window focus/minimize and process launching
│   ├── person_detector_yolov2.py  # Standalone detector with zone UI (for tuning)
│   ├── person_detector_yolo.py    # Standalone detector, no zone UI (v1 reference)
│   └── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Edit config.json

Set your paths and window titles (see Config Reference below).

### 3. Launch your tracking app independently

The orchestrator will find it by `active_window_title`.

### 4. Run the orchestrator

```bash
python scripts/orchestrator.py
# or point to a custom config:
python scripts/orchestrator.py --config path/to/config.json
```

Press `Ctrl+C` to stop.

### 5. Run in the Background (Quiet Mode)

To run the system completely in the background without a persistent console window, double-click the `run.bat` file. This script will execute hidden and launch the orchestrator quietly.

**To stop the background process:**
Double-click the `kill.bat` file to cleanly terminate the background orchestrator and the VLC video player.

## Config Reference

```jsonc
{
    // Camera device index (0 = default webcam)
    "camera_id": 0,

    "detection": {
        // YOLOv8 model size: n (nano, fastest) → x (extra-large, most accurate)
        "model": "n",
        // Minimum confidence to count a detection (0–1)
        "confidence": 0.45,
        // NMS IOU threshold (0–1)
        "iou": 0.45,
        // Run inference every N frames (higher = faster loop, less responsive)
        "detection_interval": 2,
        // Seconds without a zone hit before presence is cleared
        "presence_timeout": 2.0,
        // Consecutive zone hits required before declaring presence
        "min_hits": 2,
        // Consecutive frame-grab failures before the detector gives up
        "max_failures": 10,
        // Detection zone as fractions of frame size (0.0–1.0)
        // Legacy rectangular fallback — used only when zone_poly is absent
        "zone": { "x1": 0.20, "y1": 0.15, "x2": 0.80, "y2": 0.85 },

        // Trapezoidal detection zone — preferred over "zone" when present.
        // Four [x, y] points in normalised coords (0.0–1.0), listed in order:
        // bottom-left, bottom-right, top-right, top-left.
        // The bottom edge should be wider (close to camera) and the top edge
        // narrower (far from camera) to match the perspective of a floor zone
        // viewed by a front-facing camera.
        // Presence is triggered when a person's feet (bottom-centre of their
        // bounding box) fall inside this polygon.
        "zone_poly": [
            [0.005, 0.700],
            [0.005, 0.770],
            [0.615, 0.625],
            [0.050, 0.630]
        ]
    },

    // Substring of the tracking app's window title
    "active_window_title": "My Tracking App",

    // Full path to the idle video file (VLC launched with --loop --fullscreen)
    "idle_video_path": "C:/media/idle.mp4",

    // Substring of the idle video player's window title
    "idle_player_title": "VLC media player",

    // Seconds to wait after presence is lost before switching back to idle
    "grace_period_seconds": 3.0,

    // Seconds to sleep between window operations (helps slow apps respond)
    "switch_delay_seconds": 0.3,

    // Fade-to-black overlay that masks the hard window swap
    "transition": {
        "enabled": true,
        // Total duration in seconds (split evenly: fade-in + fade-out)
        "duration": 0.5,
        // RGB colour of the overlay — [0,0,0] = black, [255,255,255] = white flash
        "color": [0, 0, 0]
    },

    // Arc indicator shown while the detector is warming up toward min_hits
    "anticipation": {
        "enabled": true,
        // Diameter of the arc widget in pixels
        "size": 120,
        // Arc stroke width in pixels
        "thickness": 10,
        // Screen corner: tl / tr / bl / br
        "corner": "br",
        // Pixels from the screen edge
        "margin": 30,
        // Arc colour when just starting (RGB)
        "color_cold": [80, 80, 220],
        // Arc colour when about to fire (RGB)
        "color_hot": [0, 220, 80],
        // Animation steps when draining back to zero
        "drain_steps": 12,
        // Milliseconds per drain step
        "drain_ms": 20
    }
}
```

---

## Tuning the detection zone

### Rectangular zone (legacy)

Run the v2 detector standalone to visually position the zone before
committing values to config.json:

```bash
python scripts/person_detector_yolov2.py --zone-x1 0.2 --zone-y1 0.15 --zone-x2 0.8 --zone-y2 0.85
```

- Green boxes = person inside the zone (counts toward presence)
- Grey boxes = person outside the zone (ignored)
- Press `r` to reset the zone to defaults, `q` to quit

All options: `python scripts/person_detector_yolov2.py --help`

### Trapezoidal zone (recommended for front-facing cameras)

A front-facing camera looking at a floor zone introduces perspective
distortion — the zone appears wider at the bottom (close to camera) and
narrower at the top (far from camera). A rectangle cannot model this
accurately; `zone_poly` lets you define the real shape as a trapezoid.

Set `zone_poly` in `config.json` as four `[x, y]` points in normalised
(0.0–1.0) coordinates, going bottom-left → bottom-right → top-right →
top-left:

```jsonc
"zone_poly": [
    [0.005, 0.700], // bottom-left  (wide, close to camera)
    [0.005, 0.770], // bottom-right
    [0.615, 0.625], // top-right    (narrow, far from camera)
    [0.050, 0.630]  // top-left
]
```

When `zone_poly` is present it takes priority over `zone`. The hit test
checks the **feet position** (bottom-centre of the bounding box) against
the polygon using `cv2.pointPolygonTest`, so only people actually standing
inside the trapezoid trigger presence. If `zone_poly` is removed or absent,
the system falls back to the legacy rectangular `zone`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture and frame processing |
| `numpy` | Array operations |
| `ultralytics` | YOLOv8 model (weights download automatically on first run) |
| `pygetwindow` | Window title lookup and focus control |
| `tkinter` | Transition and anticipation overlays (Python stdlib, no install needed) |
