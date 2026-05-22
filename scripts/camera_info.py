#!/usr/bin/env python3
"""
Camera Info Script
Display detailed information about a specific depth camera.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
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


def get_camera_info(camera_id: int, backend: str = None) -> dict:
    """
    Get detailed information about a camera.
    
    Args:
        camera_id: Camera device ID
        backend: Optional OpenCV backend (DirectShow, MSMF, etc.)
        
    Returns:
        Dictionary containing camera information
    """
    info = {"camera_id": camera_id, "available": False, "properties": {}}
    
    if not OPENCV_AVAILABLE:
        info["error"] = "OpenCV not available"
        return info
    
    # Map backend string to OpenCV constant
    backend_map = {
        "directshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }
    
    cap_backend = None
    if backend:
        cap_backend = backend_map.get(backend.lower())
    
    if cap_backend is None:
        # Try default backends
        for cap in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap_test = cv2.VideoCapture(camera_id, cap)
            if cap_test.isOpened():
                cap_backend = cap
                cap_test.release()
                break
    
    if cap_backend is None:
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id, cap_backend)
    
    if not cap.isOpened():
        info["error"] = f"Cannot open camera {camera_id}"
        return info
    
    info["available"] = True
    
    # Get all available properties
    props = [
        ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
        ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
        ("CAP_PROP_FPS", cv2.CAP_PROP_FPS),
        ("CAP_PROP_FORMAT", cv2.CAP_PROP_FORMAT),
        ("CAP_PROP_MODE", cv2.CAP_PROP_MODE),
        ("CAP_PROP_BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
        ("CAP_PROP_CONTRAST", cv2.CAP_PROP_CONTRAST),
        ("CAP_PROP_SATURATION", cv2.CAP_PROP_SATURATION),
        ("CAP_PROP_HUE", cv2.CAP_PROP_HUE),
        ("CAP_PROP_GAIN", cv2.CAP_PROP_GAIN),
        ("CAP_PROP_EXPOSURE", cv2.CAP_PROP_EXPOSURE),
        ("CAP_PROP_CONVERT_RGB", cv2.CAP_PROP_CONVERT_RGB),
    ]
    
    for prop_name, prop_id in props:
        value = cap.get(prop_id)
        if value >= 0:  # Valid property
            info["properties"][prop_name] = value
    
    cap.release()
    return info


def print_camera_info(info: dict):
    """Pretty print camera information."""
    print(f"\n{'='*50}")
    print(f"Camera ID: {info['camera_id']}")
    print(f"{'='*50}")
    
    if "error" in info:
        print(f"ERROR: {info['error']}")
        return
    
    if not info["available"]:
        print("Camera not available")
        return
    
    print(f"Status: Available")
    print(f"\nProperties:")
    print("-" * 40)
    
    for prop, value in info["properties"].items():
        if prop in ["CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT"]:
            print(f"  {prop.replace('CAP_PROP_', '')}: {int(value)}")
        elif prop in ["CAP_PROP_FPS"]:
            print(f"  {prop.replace('CAP_PROP_', '')}: {value}")
        elif prop in ["CAP_PROP_CONVERT_RGB"]:
            print(f"  {prop.replace('CAP_PROP_', '')}: {bool(value)}")
        else:
            print(f"  {prop}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display detailed camera information"
    )
    parser.add_argument(
        "--camera-id", "-c", type=int, default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--backend", "-b", type=str, default=None,
        choices=["directshow", "msmf", "v4l2"],
        help="OpenCV backend to use"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    info = get_camera_info(args.camera_id, args.backend)
    
    if args.json:
        import json
        print(json.dumps(info, indent=2))
    else:
        print_camera_info(info)
    
    return 0 if info["available"] else 1


if __name__ == "__main__":
    sys.exit(main())