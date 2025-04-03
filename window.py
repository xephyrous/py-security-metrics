import cv2
import sys
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

from models.yolo11 import analyze_frame, draw_detections


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.cap = cv2.VideoCapture(args.input)
        if not self.cap.isOpened():
            print("Error: Couldn't open video source.")
            sys.exit()

        # Application UI layout
        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # FPS
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(self.frame_time * 1000))

        # Frame data
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_path = args.output
        self.out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

    def update_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()

        if ret:
            # keypoints_data = analyze_frame(frame)  # Get pose keypoints
            # frame_detected = draw_poses(frame, keypoints_data)  # Draw keypoints and skeleton
            keypoints_data = analyze_frame(frame)  # Get pose keypoints
            frame_detected = draw_detections(frame, keypoints_data)  # Draw keypoints and skeleton

            # Convert BGR to RGB for PyQt
            frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)

            self.out_writer.write(frame_rgb)  # Save the frame

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qimg_resized = qimg.scaled(960, 540)
            self.label.setPixmap(QPixmap.fromImage(qimg_resized))

            # Frame timing for consistent FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed_time)
            time.sleep(sleep_time)
        else:
            self.cap.release()
            self.out_writer.release()
            self.timer.stop()
            print(f"Video saved to {self.output_path}")