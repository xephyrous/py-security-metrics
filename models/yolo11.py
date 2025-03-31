import os
import cv2
from ultralytics import YOLO

# Load YOLO 11 OBB model
model = YOLO("yolo11n.pt")

def analyze_frame(frame):
    """
    Analyzes a single video frame and detects objects using YOLO.

    Args:
        frame (numpy.ndarray): The input video frame in BGR format.

    Returns:
        dict: A dictionary of detected persons with bounding boxes and confidence scores.
              Format: { "person_1": ([x1, y1, x2, y2], confidence), ... }
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = model(frame_rgb)  # Run inference on the frame
    detections = {}
    person_count = 0

    for result in results:
        if result.boxes is None:
            continue  # Skip if no detections

        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = model.names[int(cls)]  # Get class name

            if label != "person" or conf < 0.7:
                continue

            person_count += 1
            detection_label = f"person_{person_count}"
            detections[detection_label] = (box.tolist(), conf.item())

    return detections

def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame.

    Args:
        frame (numpy.ndarray): The input video frame.
        detections (dict): A dictionary of detected objects with bounding boxes and confidence scores.

    Returns:
        numpy.ndarray: The frame with drawn bounding boxes and labels.
    """
    for label, (box, score) in detections.items():
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def process_video(video_in):
    """
    Processes a video file, applies object detection, and saves the output.

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

        detections = analyze_frame(frame)
        output_frame = draw_detections(frame, detections)

        cv2.imshow(video_name, output_frame)
        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()