#!/usr/bin/env python3
"""
Performance Benchmark Script
Benchmark camera performance (FPS, latency, frame drops).

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import time
import json
from collections import deque

try:
    import cv2
except ImportError:
    print("Error: OpenCV not available")
    sys.exit(1)


def benchmark_camera(camera_id: int, duration: int = 30, warmup: int = 5) -> dict:
    """Benchmark camera performance."""
    results = {
        "camera_id": camera_id, "resolution": "1024x1024",
        "success": False, "errors": [], "metrics": {}
    }
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        results["errors"].append(f"Cannot open camera {camera_id}")
        return results
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    
    print(f"Benchmarking camera {camera_id} for {duration}s (warmup: {warmup}s)...")
    
    # Warmup phase
    warmup_start = time.time()
    while time.time() - warmup_start < warmup:
        cap.read()
    
    # Benchmark phase
    frame_times = deque(maxlen=1000)
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    while time.time() - start_time < duration:
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()
        
        if not ret:
            results["errors"].append("Frame capture failed")
            continue
        
        frame_times.append(t1 - t0)
        frame_count += 1
        
        # Progress report every 5 seconds
        if time.time() - last_report >= 5:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"  Progress: {elapsed:.0f}s - {frame_count} frames ({current_fps:.1f} fps)")
            last_report = time.time()
    
    cap.release()
    
    total_time = time.time() - start_time
    latencies = list(frame_times)
    
    if latencies:
        results["success"] = True
        results["metrics"] = {
            "total_frames": frame_count,
            "total_time_sec": round(total_time, 2),
            "avg_fps": round(frame_count / total_time, 2),
            "frame_drops": 0,  # OpenCV doesn't expose this directly
            "latency": {
                "min_ms": round(min(latencies) * 1000, 2),
                "max_ms": round(max(latencies) * 1000, 2),
                "avg_ms": round(sum(latencies) / len(latencies) * 1000, 2),
                "p50_ms": round(sorted(latencies)[len(latencies)//2] * 1000, 2),
                "p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)] * 1000, 2),
                "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)] * 1000, 2),
            }
        }
    
    return results


def compare_cameras(camera_ids: list, duration: int = 30) -> dict:
    """Benchmark and compare multiple cameras."""
    results = {"cameras": [], "comparison": {}}
    
    for cam_id in camera_ids:
        print(f"\n--- Camera {cam_id} ---")
        result = benchmark_camera(cam_id, duration)
        results["cameras"].append(result)
    
    # Comparison summary
    if all(r["success"] for r in results["cameras"]):
        results["comparison"] = {
            "best_fps": max(r["metrics"]["avg_fps"] for r in results["cameras"]),
            "lowest_latency": min(r["metrics"]["latency"]["avg_ms"] for r in results["cameras"]),
        }
    
    return results


def print_benchmark_results(results: dict):
    """Print benchmark results."""
    if "cameras" in results:
        # Comparison mode
        print("\n=== PERFORMANCE COMPARISON ===")
        for cam in results["cameras"]:
            cid = cam["camera_id"]
            m = cam.get("metrics", {})
            print(f"\nCamera {cid}:")
            print(f"  FPS: {m.get('avg_fps', 'N/A')}")
            lat = m.get("latency", {})
            print(f"  Latency: min={lat.get('min_ms','N/A')}ms, avg={lat.get('avg_ms','N/A')}ms, max={lat.get('max_ms','N/A')}ms")
    else:
        # Single camera
        print(f"\n=== BENCHMARK RESULTS: Camera {results['camera_id']} ===")
        m = results.get("metrics", {})
        print(f"Frames: {m.get('total_frames', 'N/A')}")
        print(f"Duration: {m.get('total_time_sec', 'N/A')}s")
        print(f"Average FPS: {m.get('avg_fps', 'N/A')}")
        
        lat = m.get("latency", {})
        if lat:
            print(f"\nLatency (ms):")
            print(f"  Min:    {lat.get('min_ms', 'N/A')}")
            print(f"  Avg:    {lat.get('avg_ms', 'N/A')}")
            print(f"  Max:    {lat.get('max_ms', 'N/A')}")
            print(f"  P50:    {lat.get('p50_ms', 'N/A')}")
            print(f"  P95:    {lat.get('p95_ms', 'N/A')}")
            print(f"  P99:    {lat.get('p99_ms', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark camera performance")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Benchmark duration")
    parser.add_argument("--warmup", "-w", type=int, default=5, help="Warmup duration")
    parser.add_argument("--compare", action="store_true", help="Compare multiple cameras")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.compare:
        results = compare_cameras([0, 1], args.duration)
    else:
        results = benchmark_camera(args.camera_id, args.duration, args.warmup)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_benchmark_results(results)
    
    return 0 if results.get("success", False) or results.get("cameras", [{}])[0].get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())