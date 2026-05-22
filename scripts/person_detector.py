#!/usr/bin/env python3
"""
Person Detector using Machine Learning (RGB Fallback)

Since the depth hardware is locked up, this script falls back to using the
standard RGB camera feed and applies a Machine Learning model (HOG Descriptor)
to visually detect humans in the frame.

Author: Clock-Verbal Team
Date: 2026-05-22
"""

import sys
import argparse
import cv2

def detect_person_ml(camera_id: int):
    """
    Connects to the RGB camera and uses HOG + SVM to detect humans.
    
    Args:
        camera_id: ID of the camera to connect to.
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    # Standard HD resolution for the RGB feed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Started live RGB feed from camera {camera_id}.")
    print("Initializing Machine Learning model (HOG)...")
    
    # Initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("Model loaded. Press 'q' to quit.")

    window_name = "Person Detector (ML Visual Feed)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Variables for smoothing out the detection (optional, but good for UI)
    frames_since_last_detection = 0
    presence_persistent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame slightly to improve processing speed
        frame = cv2.resize(frame, (640, 360))

        # Detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Check if anyone was detected
        if len(boxes) > 0:
            presence_persistent = True
            frames_since_last_detection = 0
        else:
            frames_since_last_detection += 1
            if frames_since_last_detection > 10:  # 10 frames of no detection resets the flag
                presence_persistent = False

        # Draw the bounding boxes
        for (x, y, w, h) in boxes:
            # Draw a green rectangle around the detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # UI Feedback
        if presence_persistent:
            # Add caption to the screen
            cv2.putText(frame, "Presence Detected!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            # Make the window border red-ish to indicate detection
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
        else:
            cv2.putText(frame, "Scanning...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Person Detection using Machine Learning (RGB)")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera ID")
    args = parser.parse_args()

    detect_person_ml(args.camera_id)

if __name__ == "__main__":
    main()
