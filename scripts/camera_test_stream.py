#!/usr/bin/env python3
"""
Camera Test Stream Script
Test streaming from a single depth camera.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import time
from pathlib import Path

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def test_camera_stream(camera_id: int, backend: str = None, 
                       duration: int = 10, display: bool = True) -> dict:
    """
    Test streaming from a camera.
    
    Args:
        camera_id: Camera device ID
        backend: Optional OpenCV backend
        duration: Test duration in seconds
        display: Whether to display frames
        
    Returns:
        Dictionary with test results
    """
    results = {
        "camera_id": camera_id,
        "success": False,
        "frames_captured": 0,
        "fps": 0.0,
        "errors": []
    }
    
    if not OPENCV_AVAILABLE:
        results["errors"].append("OpenCV not available")
        return results
    
    backend_map = {
        "directshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }
    
    cap_backend = backend_map.get(backend.lower()) if backend else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(camera_id, cap_backend)
    
    if not cap.isOpened():
        results["errors"].append(f"Cannot open camera {camera_id}")
        return results
    
    # Set to 1024x1024 for ToF sensor
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Testing camera {camera_id} at {width}x{height}")
    print(f"Duration: {duration}s (press 'q' to stop early)")
    
    start_time = time.time()
    frame_count = 0
    last_fps_update = start_time
    fps = 0.0
    
    window_name = f"Camera {camera_id} Stream"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        if not ret:
            results["errors"].append("Frame capture failed")
            break
        
        frame_count += 1
        
        # Update FPS every second
        if time.time() - last_fps_update >= 1.0:
            fps = frame_count / (time.time() - start_time)
            last_fps_update = time.time()
        
        if display:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User stopped early")
                break
        
        if elapsed >= duration:
            break
    
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    results["success"] = True
    results["frames_captured"] = frame_count
    results["fps"] = frame_count / total_time if total_time > 0 else 0
    results["resolution"] = f"{width}x{height}"
    
    return results


def print_results(results: dict):
    """Print test results."""
    print(f"\n{'='*50}")
    print("STREAM TEST RESULTS")
    print(f"{'='*50}")
    print(f"Camera ID:    {results['camera_id']}")
    print(f"Success:      {results['success']}")
    print(f"Resolution:  {results.get('resolution', 'N/A')}")
    print(f"Frames:       {results['frames_captured']}")
    print(f"Avg FPS:      {results['fps']:.2f}")
    
    if results["errors"]:
        print(f"\nErrors:")
        for err in results["errors"]:
            print(f"  - {err}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test streaming from a depth camera"
    )
    parser.add_argument(
        "--camera-id", "-c", type=int, default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--backend", "-b", type=str, default=None,
        choices=["directshow", "msmf", "v4l2"],
        help="OpenCV backend"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=10,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable frame display"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    results = test_camera_stream(
        args.camera_id,
        args.backend,
        args.duration,
        not args.no_display
    )
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print_results(results)
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())