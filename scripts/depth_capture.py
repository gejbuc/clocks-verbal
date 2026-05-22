#!/usr/bin/env python3
"""
Depth Capture Script
Capture depth frames and save to disk.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import time
import os
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: OpenCV not available")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy not available")
    sys.exit(1)


def get_timestamp():
    """Get timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


def capture_depth(camera_id: int, output_dir: str, num_frames: int = 30, 
                  display: bool = True) -> dict:
    """Capture depth frames from camera."""
    results = {"success": False, "frames_captured": 0, "saved_files": [], "errors": []}
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        results["errors"].append(f"Cannot open camera {camera_id}")
        return results
    
    # Configure for 1024x1024 ToF sensor
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    
    print(f"Capturing {num_frames} frames from camera {camera_id}")
    print(f"Output directory: {output_path}")
    
    window_name = "Depth Capture"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    timestamp = get_timestamp()
    frames = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            results["errors"].append(f"Frame {i} capture failed")
            continue
        
        frames.append(frame)
        
        if display:
            cv2.putText(frame, f"Frame: {i+1}/{num_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User stopped early")
                break
    
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Save frames
    for i, frame in enumerate(frames):
        filename = f"depth_{timestamp}_cam{camera_id}_frame{i:04d}.png"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), frame)
        results["saved_files"].append(str(filepath))
    
    results["success"] = True
    results["frames_captured"] = len(frames)
    
    # Also save as video if multiple frames
    if len(frames) > 1:
        video_path = output_path / f"depth_{timestamp}_cam{camera_id}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1024, 1024))
        for frame in frames:
            out.write(frame)
        out.release()
        results["saved_files"].append(str(video_path))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Capture depth frames to disk")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--output", "-o", type=str, default="captures", help="Output directory")
    parser.add_argument("--frames", "-f", type=int, default=30, help="Number of frames")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    results = capture_depth(args.camera_id, args.output, args.frames, not args.no_display)
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print(f"\n=== CAPTURE RESULTS ===")
        print(f"Frames captured: {results['frames_captured']}")
        print(f"Files saved: {len(results['saved_files'])}")
        for f in results["saved_files"]:
            print(f"  - {f}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())