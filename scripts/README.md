# Scripts

Python scripts for the Clocks Verbal presence detection system, plus legacy
depth camera testing utilities.

## Presence Detection (main system)

| Script | Description |
|--------|-------------|
| `orchestrator.py` | Main entry point — loads config, runs detector, manages window switching |
| `anticipation.py` | Arc overlay — warmup progress indicator shown before presence fires |
| `transition.py` | Fade-to-black overlay — masks the hard window swap |
| `person_detector_yolov2.py` | YOLOv8 detector with visual detection zone overlay (recommended for tuning) |
| `person_detector_yolo.py` | YOLOv8 detector, no zone UI (v1 reference) |
| `window_manager.py` | Window focus/minimize and process launching via `pygetwindow` |

## Depth Camera Utilities (legacy)

| Script | Description |
|--------|-------------|
| `camera_discovery.py` | Discover and list connected depth cameras |
| `camera_info.py` | Display detailed info about a specific camera |
| `camera_test_stream.py` | Test streaming from a single camera |
| `camera_dual_stream.py` | Test simultaneous streaming from both cameras |
| `depth_capture.py` | Capture depth frames and save to disk |
| `calibration_check.py` | Verify camera calibration parameters |
| `performance_benchmark.py` | Benchmark camera performance (FPS, latency) |
| `visualization.py` | Visualize depth data with colour mapping |
| `device_validation.py` | Comprehensive validation test suite |

## Requirements

```bash
pip install -r requirements.txt
```

## Running the orchestrator

```bash
python orchestrator.py                          # uses ../config.json by default
python orchestrator.py --config my_config.json  # custom config path
```

## Tuning the detection zone

Run the v2 detector standalone to visually adjust zone boundaries before
committing them to config.json:

```bash
python person_detector_yolov2.py --zone-x1 0.2 --zone-y1 0.15 --zone-x2 0.8 --zone-y2 0.85
```

All detector options:

```bash
python person_detector_yolov2.py --help
```
