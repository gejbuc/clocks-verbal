# Depth Camera Testing Scripts

This directory contains testing and validation scripts for the Depth Camera A (1-Megapixel ToF sensor based on HoloLens 2 technology).

## Camera Specifications

- **Sensor Type**: Time-of-Flight (ToF)
- **Resolution**: 1024 × 1024 (1 Megapixel)
- **Technology**: HoloLens 2 depth sensor
- **Camera Count**: 2 units

## Available Scripts

| Script | Description |
|--------|-------------|
| `camera_discovery.py` | Discover and list all connected depth cameras |
| `camera_info.py` | Display detailed information about a specific camera |
| `camera_test_stream.py` | Test streaming from a single camera |
| `camera_dual_stream.py` | Test simultaneous streaming from both cameras |
| `depth_capture.py` | Capture depth frames and save to disk |
| `calibration_check.py` | Verify camera calibration parameters |
| `performance_benchmark.py` | Benchmark camera performance (FPS, latency) |
| `visualization.py` | Visualize depth data with color mapping |
| `device_validation.py` | Comprehensive validation test suite |

## Requirements

Install required dependencies:

```bash
pip install opencv-python numpy pyrealsense2 pyk4a
```

## Quick Start

### 1. Discover cameras
```bash
python camera_discovery.py
```

### 2. Test single camera stream
```bash
python camera_test_stream.py --camera-id 0
```

### 3. Test dual camera stream
```bash
python camera_dual_stream.py
```

### 4. Run full validation
```bash
python device_validation.py --camera-a 0 --camera-b 1
```

## Notes

- Camera IDs may vary depending on system configuration
- Some scripts require specific SDKs (RealSense, Azure Kinect)
- Run with `--help` for additional options on each script