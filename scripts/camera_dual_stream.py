#!/usr/bin/env python3
"""
Camera Dual Stream Script
Test simultaneous streaming from both depth cameras.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import time
import threading
from queue import Queue

try:
    import cv2
except ImportError:
    print("Error: OpenCV not available")
    sys.exit(1)


def capture_camera(camera_id: int, queue: Queue, stop_event: threading.Event):
    """Capture frames from a camera into a queue."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        queue.put(("error", f"Cannot open camera {camera_id}"))
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            queue.put(("error", f"Camera {camera_id} frame error"))
            break
        queue.put(("frame", frame))
    
    cap.release()


def test_dual_stream(camera_a: int = 0, camera_b: int = 1,
                     duration: int = 10, display: bool = True) -> dict:
    """Test simultaneous streaming from two cameras."""
    results = {
        "camera_a": camera_a, "camera_b": camera_b, "success": False,
        "camera_a_frames": 0, "camera_b_frames": 0,
        "camera_a_fps": 0.0, "camera_b_fps": 0.0, "sync_test": False, "errors": []
    }
    
    queue_a, queue_b = Queue(maxsize=2), Queue(maxsize=2)
    stop_event = threading.Event()
    
    thread_a = threading.Thread(target=capture_camera, args=(camera_a, queue_a, stop_event))
    thread_b = threading.Thread(target=capture_camera, args=(camera_b, queue_b, stop_event))
    
    print(f"Dual stream: Camera A={camera_a}, B={camera_b}, Duration={duration}s")
    
    thread_a.start()
    thread_b.start()
    
    start_time = time.time()
    frames_a = frames_b = 0
    
    if display:
        cv2.namedWindow("Dual Camera Stream", cv2.WINDOW_NORMAL)
    
    try:
        while time.time() - start_time < duration:
            f_a = f_b = None
            try:
                _, f_a = queue_a.get_nowait()
            except:
                pass
            try:
                _, f_b = queue_b.get_nowait()
            except:
                pass
            
            elapsed = time.time() - start_time
            
            if f_a is not None:
                frames_a += 1
            if f_b is not None:
                frames_b += 1
            
            if display and (f_a is not None or f_b is not None):
                combined = cv2.hconcat([f_a, f_b]) if f_a is not None and f_b is not None else (f_a or f_b)
                fps_a = frames_a / elapsed if elapsed > 0 else 0
                fps_b = frames_b / elapsed if elapsed > 0 else 0
                label = f"Cam A: {fps_a:.1f}fps | Cam B: {fps_b:.1f}fps"
                cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Dual Camera Stream", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        stop_event.set()
        thread_a.join(timeout=2)
        thread_b.join(timeout=2)
        if display:
            cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    results["success"] = True
    results["camera_a_frames"] = frames_a
    results["camera_b_frames"] = frames_b
    results["camera_a_fps"] = frames_a / total_time if total_time > 0 else 0
    results["camera_b_fps"] = frames_b / total_time if total_time > 0 else 0
    results["sync_test"] = abs(results["camera_a_fps"] - results["camera_b_fps"]) < 2.0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test dual camera streaming")
    parser.add_argument("--camera-a", "-a", type=int, default=0, help="First camera ID")
    parser.add_argument("--camera-b", "-b", type=int, default=1, help="Second camera ID")
    parser.add_argument("--duration", "-d", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    results = test_dual_stream(args.camera_a, args.camera_b, args.duration, not args.no_display)
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print(f"\n=== DUAL STREAM RESULTS ===")
        print(f"Camera A: {results['camera_a_frames']} frames @ {results['camera_a_fps']:.1f}fps")
        print(f"Camera B: {results['camera_b_frames']} frames @ {results['camera_b_fps']:.1f}fps")
        print(f"Sync Test: {'PASS' if results['sync_test'] else 'FAIL'}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())