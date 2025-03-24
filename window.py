import sys
import cv2
import torch
import torchvision
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class Window(QMainWindow):
    def __init__(self, video_source=0, output_path="output_video.mp4"):
        super().__init__()
        self.setWindowTitle("PyQt Video Display")
        self.setGeometry(100, 100, 800, 600)

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Error: Couldn't open video source.")
            sys.exit()

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_path = output_path
        self.out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

    def analyze_frame(self, frame):
        img = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img)[0]

        detections = {}
        for i in range(len(outputs["boxes"])):
            label_index = int(outputs["labels"][i].item())
            if label_index != 1:
                continue

            box = outputs["boxes"][i].tolist()
            mask = outputs["masks"][i][0].mul(255).byte().cpu().numpy()
            score = outputs["scores"][i].item()

            if score > 0.7:
                detections[f"person_{i + 1}"] = (box, mask, score)
        return detections

    def draw_detections(self, frame, detections):
        height, width, _ = frame.shape
        left_half_rect = (0, 0, width // 2, height)
        cv2.rectangle(frame, (left_half_rect[0], left_half_rect[1]), (left_half_rect[2], left_half_rect[3]),
                      (255, 0, 0), 2)

        for label, (box, mask, score) in detections.items():
            x1, y1, x2, y2 = map(int, box)
            mask_cropped = mask[y1:y2, x1:x2]
            mask_colored = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)
            frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1, mask_colored, 0.5, 0)

            box_color = (0, 0, 0)
            if x1 < width // 2:
                box_color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.analyze_frame(frame_rgb)
            frame_detected = self.draw_detections(frame_rgb, detections)

            self.out_writer.write(cv2.cvtColor(frame_detected, cv2.COLOR_RGB2BGR))

            h, w, ch = frame_detected.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_detected.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.cap.release()
            self.out_writer.release()
            self.timer.stop()
            print(f"Video saved to {self.output_path}")
            os.system(f"start {self.output_path}" if os.name == "nt" else f"xdg-open {self.output_path}")
