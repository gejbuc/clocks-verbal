#!/usr/bin/env python3
"""
Device Validation Script - Simple Version
Quick validation test for depth cameras.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import json
import time
import cv2


def test_connection(camera_id: int) -> dict:
    """Test if camera can be opened."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    result = {"test": "connection", "passed": False, "message": ""}
    
    if cap.isOpened():
        result["passed"] = True
        result["message"] = f"Camera {camera_id} connected"
        cap.release()
    else:
        result["message"] = f"Cannot connect to camera {camera_id}"
    
    return result


def test_resolution(camera_id: int, expected_w: int = 1024, expected_h: int = 1024) -> dict:
    """Test camera resolution."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    result = {"test": "resolution", "passed": False, "message": ""}
    
    if not cap.isOpened():
        result["message"] = "Cannot open camera"
        return result
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_h)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if width == expected_w and height == expected_h:
        result["passed"] = True
        result["message"] = f"{width}x{height}"
    else:
        result["message"] = f"Got {width}x{height}, expected {expected_w}x{expected_h}"
    
    return result


def test_streaming(camera_id: int, duration: int = 5) -> dict:
    """Test streaming."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    result = {"test": "streaming", "passed": False, "message": ""}
    
    if not cap.isOpened():
        result["message"] = "Cannot open camera"
        return result
    
    frames = 0
    start = time.time()
    
    while time.time() - start < duration:
        if cap.read()[0]:
            frames += 1
    
    cap.release()
    
    fps = frames / duration
    if fps > 0:
        result["passed"] = True
        result["message"] = f"{frames} frames ({fps:.1f} fps)"
    else:
        result["message"] = "No frames captured"
    
    return result


def test_frame_quality(camera_id: int, num_frames: int = 10) -> dict:
    """Test frame quality."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    result = {"test": "quality", "passed": False, "message": ""}
    
    if not cap.isOpened():
        result["message"] = "Cannot open camera"
        return result
    
    valid = sum(1 for _ in range(num_frames) if cap.read()[0])
    cap.release()
    
    if valid == num_frames:
        result["passed"] = True
        result["message"] = f"All {num_frames} frames valid"
    else:
        result["message"] = f"{valid}/{num_frames} valid"
    
    return result


def run_validation(camera_id: int = 0) -> dict:
    """Run validation tests."""
    results = {"camera_id": camera_id, "tests": [], "passed": False}
    
    tests = [test_connection, test_resolution, test_streaming, test_frame_quality]
    
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test(camera_id)
        results["tests"].append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status}: {result['message']}")
    
    results["passed"] = all(t["passed"] for t in results["tests"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate depth camera")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    results = run_validation(args.camera_id)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*40}")
        print(f"OVERALL: {'PASSED' if results['passed'] else 'FAILED'}")
        print(f"{'='*40}")
    
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())