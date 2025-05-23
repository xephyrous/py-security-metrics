import os
import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

model = None

# if torch.cuda.is_available():
#     model.to('cuda')
#     print("On cuda")
# else:
#     print("On cpu.")

running_speed = 0
frame_speed = 0
frame_total = 0
tile_total = 0

temp_total = 0
temp_speed = 0


def get_stats():
    """
    Functions to get all stats of analysis

    Args:

    Returns:
        tuple: A tuple containing all stats:
              Format: (Total Time, Frames analyzed, Tiles analyzed, Frame Average, Tile Average)
    """
    global running_speed, frame_speed, frame_total, tile_total

    return (running_speed, frame_total, tile_total, 0 if frame_total == 0 else frame_speed/frame_total, 0 if tile_total == 0 else running_speed/tile_total)

def analyze_tile(tile, offset_x, offset_y, debug_window = False):
    """
    Analyzes a single tile of a video frame and detects objects using YOLO.

    Args:
        tile: tile to analyze.
        offset_x: x offset of tile.
        offset_y: y offset of tile.
        debug_window: whether to display debug window.

    Returns:
        dict: A dictionary of detected persons with bounding boxes and confidence scores.
              Format: [ { "person_1": ([x1, y1, x2, y2], confidence), ... } ]
    """

    global running_speed
    global tile_total

    global frame_speed
    global frame_total

    global temp_speed
    global temp_total

    if debug_window:
        labeled_tile = tile.copy()
        cv2.putText(
            labeled_tile,
            f"Analyzing Tile @ ({offset_x}, {offset_y})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        cv2.putText(
            labeled_tile,
            f"Average Tile Computation Time: {'N/A' if tile_total == 0 else round(running_speed/tile_total, 2)}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
        cv2.putText(
            labeled_tile,
            f"Average Frame Computation Time: {'N/A' if frame_total == 0 else round(frame_speed/frame_total, 2)}ms",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
        cv2.imshow("Tile", labeled_tile)
        cv2.waitKey(1)

    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    results = model(tile_rgb, verbose=False)

    running_speed += results[0].speed['inference']
    tile_total += 1
    temp_speed += results[0].speed['inference']
    temp_total += 1

    detections = {}

    person_count = 0
    for result in results:
        if result.boxes is None:
            continue

        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = model.names[int(cls)]  # Get class name

            if label != "person" or conf < 0.3:
                continue

            person_count += 1
            detection_label = f"person_{person_count}"

            # Adjust coordinates back to original image
            x1, y1, x2, y2 = box.tolist()
            adjusted_box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]

            detections[detection_label] = (adjusted_box, conf.item())

    return detections

def analyze_frame(frame, tile_size=640, overlap=0, iou_thresh=0.5, debug_window=False):
    """
    Analyzes a single video frame and detects objects using YOLO.

    Args:
        frame (numpy.ndarray): The input video frame in BGR format.
        tile_size: size of tile to split, defaults to 640.
        overlap: overlap of tile, defaults to 0.
        iou_thresh: NMS Iou thresh, defaults to 0.5.
        debug_window: whether to display debug window.

    Returns:
        dict: A dictionary of detected persons with bounding boxes and confidence scores.
              Format: { "person_1": ([x1, y1, x2, y2], confidence), ... }
    """
    detections = []
    height, width, _ = frame.shape
    step = tile_size - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            if x_end - x < tile_size and x_end == width:
                x = max(0, width - tile_size)
                x_end = width
            if y_end - y < tile_size and y_end == height:
                y = max(0, height - tile_size)
                y_end = height

            tile = frame[y:y_end, x:x_end].copy()

            tile_detections = analyze_tile(tile, x, y, debug_window=debug_window)

            for box, conf in tile_detections.values():
                detections.append((box, conf))

    global running_speed
    global frame_speed
    global temp_total
    global temp_speed

    global frame_total
    frame_total += 1

    frame_speed += temp_speed

    temp_total = 0
    temp_speed = 0

    if not detections:
        return {}

    # Apply NMS
    boxes = torch.tensor([d[0] for d in detections])
    scores = torch.tensor([d[1] for d in detections])
    keep_indices = nms(boxes, scores, iou_thresh)

    final_detections = {}
    for i, idx in enumerate(keep_indices):
        label = f"person_{i+1}"

        box = boxes[idx].tolist()

        width = box[2] - box[0]

        box[0] += width/4
        box[2] -= width/4

        box[1] = box[3] - ((box[3] - box[1]) / 5)

        final_detections[label] = (boxes[idx].tolist(), scores[idx].item(), box)

    return final_detections

