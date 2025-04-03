import os
import cv2
import numpy as np
from ultralytics import YOLO

model = None

# Define keypoint connections for the skeleton
LIMB_CONNECTIONS = [
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Shoulder to hip
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6),  # Connect both shoulders
    (11, 12),  # Connect both hips
    (0, 5), (0, 6),  # Connect neck to shoulders
]

def analyze_frame(frame):
    """
    Analyzes a single video frame and extracts pose keypoints.

    Args:
        frame (numpy.ndarray): The input video frame in BGR format.

    Returns:
        list of tuple: A list of detected keypoints and their confidence scores.
                       Each entry contains (keypoints, confidence), where:
                       - keypoints is a (17,2) numpy array with (x, y) positions.
                       - confidence is a (17,) numpy array with confidence scores.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = model(frame_rgb)  # Run inference on the frame
    keypoints_data = []

    for result in results:
        # No poses detected
        if result.keypoints.xy.numel() == 0:
            print("No keypoints found in the frame")
            continue

        keypoints = result.keypoints.xy[0].cpu().numpy()  # Extract keypoints
        confidence = result.keypoints.conf[0].cpu().numpy()  # Extract confidence scores

        keypoints_data.append((keypoints, confidence))

    return keypoints_data

def draw_poses(frame, keypoints_data):
    """
    Draws human pose keypoints and skeletons on a video frame.

    Args:
        frame (numpy.ndarray): The input video frame.
        keypoints_data (list of tuple): List of keypoints and confidence scores.

    Returns:
        numpy.ndarray: The frame with drawn keypoints and skeletons.
    """
    if not keypoints_data:
        return frame

    for keypoints, confidence in keypoints_data:
        for i, (x, y) in enumerate(keypoints):
            # Ignore low confidence keypoints
            if confidence[i] < 0.3 or (x == 0 and y == 0):
                continue

            # Draw keypoints
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw limbs
        for (p1, p2) in LIMB_CONNECTIONS:
            if confidence[p1] > 0.3 and confidence[p2] > 0.3:
                x1, y1 = map(int, keypoints[p1])
                x2, y2 = map(int, keypoints[p2])
                if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw skeleton

    return frame

def process_video(video_in):
    """
    Processes a video file, applies pose detection, and saves the output.

    Args:
        video_in (str): Path to the input video file.

    Returns:
        None
    """
    video_name = os.path.splitext(os.path.basename(video_in))[0]
    output_folder = os.path.join("output", video_name)
    output_video_path = os.path.join(output_folder, f"{video_name}_processed.mp4")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file '{video_in}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints_data = analyze_frame(frame)
        output_frame = draw_poses(frame, keypoints_data)

        cv2.imshow(video_name, output_frame)
        out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
