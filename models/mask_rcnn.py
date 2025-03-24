import os

import cv2
import numpy as np
import torch
import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def analyze_frame(frame):
    img = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)[0]

    detections = {}
    person_count = 0  # Counter for people

    for i in range(len(outputs["boxes"])):
        label_index = int(outputs["labels"][i].item())  # Get class index
        if label_index != 1:  # Only count persons (label == 1)
            continue

        person_count += 1  # Increment only for persons

        label = f"person_{person_count}"  # Label indexed correctly
        box = outputs["boxes"][i].tolist()
        mask = outputs["masks"][i][0].mul(255).byte().cpu().numpy()  # Keep as NumPy array
        score = outputs["scores"][i].item()

        if score > 0.7:
            detections[label] = (box, mask, score)

    return detections

def resize_with_padding(image, target_size=(480, 480)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create a blank black image of target size
    padded = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    # Center the resized mask in the black image
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def draw_detections(frame, detections):
    for label, (box, mask, score) in detections.items():
        x1, y1, x2, y2 = map(int, box)

        # Extract the mask region corresponding to the bounding box
        mask_cropped = mask[y1:y2, x1:x2]

        # Convert grayscale mask to BGR for overlay
        mask_colored = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)

        # Overlay mask with transparency
        frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1, mask_colored, 0.5, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def process_video(video_in, output):
    video_name = os.path.splitext(os.path.basename(video_in))[0]
    output_folder = os.path.join("output", video_name)
    output_video_path = os.path.join(output_folder, f"{video_name}_processed.mp4")
    output_masks_folder = os.path.join(output_folder, "masks")

    os.makedirs(output_masks_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file '{video_in}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    person_videos = {}

    OUTPUT_SIZE = (480, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = analyze_frame(frame)

        outputFrame = draw_detections(frame, output)

        cv2.imshow(video_name, outputFrame)

        out.write(outputFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    for writer in person_videos.values():
        writer.release()