#!/usr/bin/env python3
"""
Depth Visualization Script
Visualize depth data with color mapping.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse

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


# Colormap presets for depth visualization
COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "ocean": cv2.COLORMAP_OCEAN,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "winter": cv2.COLORMAP_WINTER,
    "autumn": cv2.COLORMAP_AUTUMN,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
}


def apply_colormap(depth_frame, colormap="turbo", min_depth=0.0, max_depth=10.0):
    """Apply colormap to depth frame."""
    # Normalize to 0-255 range
    depth_norm = np.clip(depth_frame, min_depth, max_depth)
    depth_norm = ((depth_norm - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # Apply colormap
    cmap = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_TURBO)
    colored = cv2.applyColorMap(depth_norm, cmap)
    
    return colored


def visualize_depth(camera_id: int, colormap: str = "turbo", 
                    min_depth: float = 0.0, max_depth: float = 10.0) -> dict:
    """Visualize depth frames from camera with colormap."""
    results = {"success": False, "errors": []}
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        results["errors"].append(f"Cannot open camera {camera_id}")
        return results
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    
    print(f"Visualizing depth from camera {camera_id}")
    print(f"Colormap: {colormap}, Depth range: {min_depth}m - {max_depth}m")
    print("Press 'q' to quit, 'h' to toggle colormap")
    
    window_name = f"Depth Visualization - Camera {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    cmap_keys = list(COLORMAPS.keys())
    current_cmap_idx = cmap_keys.index(colormap.lower()) if colormap.lower() in cmap_keys else 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            results["errors"].append("Frame capture failed")
            break
        
        # Apply colormap
        colored = apply_colormap(frame, cmap_keys[current_cmap_idx], min_depth, max_depth)
        
        # Add info overlay
        info_text = f"Colormap: {cmap_keys[current_cmap_idx]} | Range: {min_depth}-{max_depth}m"
        cv2.putText(colored, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, colored)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            current_cmap_idx = (current_cmap_idx + 1) % len(cmap_keys)
            print(f"Switched to: {cmap_keys[current_cmap_idx]}")
        elif key in [ord(str(i)) for i in range(10)]:
            # Number keys for quick depth range presets
            max_depth = float(key - ord('0') + 1)
            print(f"Changed max depth to: {max_depth}m")
    
    cap.release()
    cv2.destroyAllWindows()
    results["success"] = True
    
    return results


def visualize_from_file(filepath: str, colormap: str = "turbo",
                         min_depth: float = 0.0, max_depth: float = 10.0):
    """Visualize depth from saved file."""
    frame = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        print(f"Error: Cannot load file {filepath}")
        return
    
    colored = apply_colormap(frame, colormap, min_depth, max_depth)
    
    cv2.imshow("Depth Visualization", colored)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize depth data")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--colormap", "-m", type=str, default="turbo",
                       choices=list(COLORMAPS.keys()), help="Colormap")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Min depth (m)")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Max depth (m)")
    parser.add_argument("--file", "-f", type=str, default=None, help="Load from file instead")
    args = parser.parse_args()
    
    if args.file:
        visualize_from_file(args.file, args.colormap, args.min_depth, args.max_depth)
    else:
        results = visualize_depth(args.camera_id, args.colormap, args.min_depth, args.max_depth)
        if not results["success"]:
            for err in results["errors"]:
                print(f"Error: {err}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())