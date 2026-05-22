#!/usr/bin/env python3
"""
Camera Discovery Script
Discovers and lists all connected depth cameras on the system.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Try importing optional dependencies
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


class CameraInfo:
    """Container for camera device information."""
    def __init__(self, backend: str, device_id: int, name: str, properties: Dict[str, Any]):
        self.backend = backend
        self.device_id = device_id
        self.name = name
        self.properties = properties
    
    def __repr__(self):
        return f"CameraInfo(backend={self.backend}, id={self.device_id}, name={self.name})"


def discover_opencv_cameras(max_devices: int = 10) -> List[CameraInfo]:
    """Discover cameras using OpenCV backends."""
    cameras = []
    
    if not OPENCV_AVAILABLE:
        return cameras
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"),
    ]
    
    for backend, backend_name in backends:
        for device_id in range(max_devices):
            try:
                cap = cv2.VideoCapture(device_id, backend)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_name = f"Camera {device_id} ({backend_name})"
                    properties = {
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "backend": backend_name,
                        "device_id": device_id
                    }
                    
                    cameras.append(CameraInfo(
                        backend=backend_name,
                        device_id=device_id,
                        name=camera_name,
                        properties=properties
                    ))
                    print(f"  Found: {camera_name} @ {width}x{height}")
                    cap.release()
                else:
                    cap.release()
            except Exception:
                continue
    
    return cameras


def discover_realsense() -> List[CameraInfo]:
    """Discover Intel RealSense depth cameras."""
    cameras = []
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            firmware = dev.get_info(rs.camera_info.firmware_version)
            print(f"  Found: RealSense - {name}")
            print(f"    Serial: {serial}, Firmware: {firmware}")
            
            properties = {
                "serial": serial,
                "firmware": firmware,
                "product_line": dev.get_info(rs.camera_info.product_line),
            }
            cameras.append(CameraInfo("RealSense", 0, name, properties))
    except ImportError:
        print("  pyrealsense2 not installed")
    except Exception as e:
        print(f"  RealSense error: {e}")
    
    return cameras


def discover_azure_kinect() -> List[CameraInfo]:
    """Discover Azure Kinect DK depth cameras."""
    cameras = []
    try:
        import pyk4a
        devices = pyk4a.Device.enumerate()
        
        for idx, serial in enumerate(devices):
            print(f"  Found: Azure Kinect {idx} (Serial: {serial})")
            properties = {"serial": serial, "index": idx}
            cameras.append(CameraInfo("AzureKinect", idx, f"Azure Kinect {idx}", properties))
    except ImportError:
        print("  pyk4a not installed")
    except Exception as e:
        print(f"  Azure Kinect error: {e}")
    
    return cameras


def discover_all() -> List[CameraInfo]:
    """Discover all available depth cameras."""
    all_cameras = []
    
    print("\n=== DEPTH CAMERA DISCOVERY ===\n")
    
    print("[*] Checking Intel RealSense...")
    all_cameras.extend(discover_realsense())
    
    print("\n[*] Checking Azure Kinect...")
    all_cameras.extend(discover_azure_kinect())
    
    print("\n[*] Checking OpenCV cameras...")
    all_cameras.extend(discover_opencv_cameras())
    
    print(f"\n=== SUMMARY: {len(all_cameras)} camera(s) found ===\n")
    
    return all_cameras


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Discover connected depth cameras")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--max-devices", type=int, default=10, help="Max devices to check")
    args = parser.parse_args()
    
    cameras = discover_all()
    
    if args.json:
        output = [
            {"backend": c.backend, "device_id": c.device_id, "name": c.name, "properties": c.properties}
            for c in cameras
        ]
        print(json.dumps(output, indent=2))
    
    return 0 if cameras else 1


if __name__ == "__main__":
    sys.exit(main())