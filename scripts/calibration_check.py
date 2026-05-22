#!/usr/bin/env python3
"""
Calibration Check Script
Verify camera calibration parameters.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import json
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV not available")
    sys.exit(1)


# Default calibration parameters for HoloLens 2 ToF sensor (1024x1024)
DEFAULT_CALIBRATION = {
    "camera_a": {
        "resolution": [1024, 1024],
        "fx": 512.0, "fy": 512.0,
        "cx": 512.0, "cy": 512.0,
        "k1": 0.0, "k2": 0.0, "k3": 0.0,
        "p1": 0.0, "p2": 0.0,
        "depth_scale": 0.001,
        "baseline": 0.0
    },
    "camera_b": {
        "resolution": [1024, 1024],
        "fx": 512.0, "fy": 512.0,
        "cx": 512.0, "cy": 512.0,
        "k1": 0.0, "k2": 0.0, "k3": 0.0,
        "p1": 0.0, "p2": 0.0,
        "depth_scale": 0.001,
        "baseline": 0.075  # Typical baseline for stereo ToF
    }
}


def load_calibration(filepath: str) -> dict:
    """Load calibration from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Calibration file not found: {filepath}")
        print("Using default parameters")
        return DEFAULT_CALIBRATION
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in calibration file: {e}")
        return DEFAULT_CALIBRATION


def verify_calibration(calibration: dict, camera_id: str = "camera_a") -> dict:
    """Verify calibration parameters are valid."""
    results = {
        "camera_id": camera_id,
        "valid": True,
        "checks": [],
        "warnings": [],
        "errors": []
    }
    
    if camera_id not in calibration:
        results["errors"].append(f"Camera '{camera_id}' not found in calibration")
        results["valid"] = False
        return results
    
    cam = calibration[camera_id]
    
    # Check resolution
    res = cam.get("resolution", [0, 0])
    if res[0] == 1024 and res[1] == 1024:
        results["checks"].append(f"Resolution: {res[0]}x{res[1]} ✓")
    else:
        results["warnings"].append(f"Resolution: {res[0]}x{res[1]} (expected 1024x1024)")
    
    # Check focal length
    fx, fy = cam.get("fx", 0), cam.get("fy", 0)
    if fx > 0 and fy > 0:
        results["checks"].append(f"Focal length: fx={fx}, fy={fy} ✓")
    else:
        results["errors"].append("Invalid focal length")
        results["valid"] = False
    
    # Check principal point
    cx, cy = cam.get("cx", 0), cam.get("cy", 0)
    if cx > 0 and cy > 0:
        results["checks"].append(f"Principal point: cx={cx}, cy={cy} ✓")
    else:
        results["warnings"].append("Principal point may be invalid")
    
    # Check distortion coefficients
    for coef in ["k1", "k2", "k3", "p1", "p2"]:
        if coef in cam:
            results["checks"].append(f"Distortion {coef}: {cam[coef]} ✓")
    
    # Check depth scale
    ds = cam.get("depth_scale", 0)
    if ds > 0:
        results["checks"].append(f"Depth scale: {ds} ✓")
    else:
        results["warnings"].append("Depth scale not set or invalid")
    
    return results


def check_stereo_calibration(calibration: dict) -> dict:
    """Check stereo calibration between cameras."""
    results = {
        "stereo_valid": True,
        "baseline": 0.0,
        "checks": [],
        "errors": []
    }
    
    # Check baseline
    if "camera_b" in calibration and "baseline" in calibration["camera_b"]:
        baseline = calibration["camera_b"]["baseline"]
        results["baseline"] = baseline
        if baseline > 0:
            results["checks"].append(f"Stereo baseline: {baseline}m ✓")
        else:
            results["errors"].append("Invalid stereo baseline")
            results["stereo_valid"] = False
    else:
        results["warnings"].append("No baseline information available")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify camera calibration")
    parser.add_argument("--calibration", "-c", type=str, default=None,
                       help="Calibration JSON file path")
    parser.add_argument("--camera-a", action="store_true", help="Check camera A")
    parser.add_argument("--camera-b", action="store_true", help="Check camera B")
    parser.add_argument("--stereo", action="store_true", help="Check stereo calibration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    # Use default if no file provided
    calibration = load_calibration(args.calibration) if args.calibration else DEFAULT_CALIBRATION
    
    all_results = {"calibration": calibration}
    
    if args.camera_a or not (args.camera_a or args.camera_b or args.stereo):
        all_results["camera_a"] = verify_calibration(calibration, "camera_a")
    
    if args.camera_b or not (args.camera_a or args.camera_b or args.stereo):
        all_results["camera_b"] = verify_calibration(calibration, "camera_b")
    
    if args.stereo or not (args.camera_a or args.camera_b):
        all_results["stereo"] = check_stereo_calibration(calibration)
    
    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n=== CALIBRATION CHECK RESULTS ===")
        for key, val in all_results.items():
            if key == "calibration":
                continue
            print(f"\n{key.upper()}:")
            if "valid" in val:
                print(f"  Valid: {val['valid']}")
            for check in val.get("checks", []):
                print(f"  {check}")
            for warn in val.get("warnings", []):
                print(f"  WARNING: {warn}")
            for err in val.get("errors", []):
                print(f"  ERROR: {err}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())