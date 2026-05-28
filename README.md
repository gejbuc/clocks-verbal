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
YOLOv8 detector  ──── detection zone (ROI) ────►  presence_on / presence_off
                                                          │
                                              ┌───────────┴───────────┐
                                           IDLE state            ACTIVE state
                                        (idle video)         (tracking app window)
```

- The **tracking app** is launched independently — the orchestrator finds it
  by window title and brings it forward on presence.
- The **idle video** is launched and managed by the orchestrator. It plays
  on a loop whenever no presence is detected.
- If the tracking window is not found when presence is detected, a warning
  is logged and the system stays on idle until the window appears.

## Project Structure

```
clocks-verbal/
├── config.json              # Runtime configuration (see below)
├── scripts/
│   ├── orchestrator.py      # Main entry point — ties detector + window manager together
│   ├── person_detector_yolo.py    # Standalone YOLOv8 detector (v1)
│   ├── person_detector_yolov2.py  # Standalone YOLOv8 detector with detection zone UI (v2)
│   ├── window_manager.py    # Window focus/minimize and process launching
│   └── requirements.txt     # Python dependencies
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
        // Only detections overlapping this rectangle count toward presence
        "zone": { "x1": 0.20, "y1": 0.15, "x2": 0.80, "y2": 0.85 }
    },

    // Substring of the tracking app's window title
    // The orchestrator searches for this and brings it forward on presence
    "active_window_title": "My Tracking App",

    // Full path to the idle video file
    "idle_video_path": "C:/media/idle.mp4",

    // Substring of the idle video player's window title (used to find/focus it)
    "idle_player_title": "VLC media player",

    // Seconds to wait after presence is lost before switching back to idle
    "grace_period_seconds": 3.0,

    // Seconds to sleep between window operations (helps slow apps respond)
    "switch_delay_seconds": 0.3
}
```

## Standalone Detectors

The detector scripts can be run independently for testing and tuning:

```bash
# Basic YOLOv8 detector
python scripts/person_detector_yolo.py --camera-id 0 --model n

# Detector with visual detection zone overlay
python scripts/person_detector_yolov2.py --zone-x1 0.2 --zone-x2 0.8
```

Both support `--help` for the full list of options.

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture and frame processing |
| `numpy` | Array operations |
| `ultralytics` | YOLOv8 model (auto-downloads weights on first run) |
| `pygetwindow` | Window title lookup and focus control |
