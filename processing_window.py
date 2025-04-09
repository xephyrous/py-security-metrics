import json
import os.path

import cv2
import sys
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

from models.yolo11 import analyze_frame, draw_detections


class ProcessingWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.proc_wind = True if args.proc_wind else False

        # Zone loading
        if args.zones and not os.path.exists(args.zones):
            raise Exception("Incorrect zones file path!")

        if args.zones:
            with open(args.zones) as file:
                obj = json.loads(file.read())

                if not all(entry in obj for entry in ["name", "description", "zones"]):
                    raise Exception("Missing required member in zone file!")

                self.zones = obj["zones"]
        else:
            self.zones = []

        # Video loading
        self.cap = cv2.VideoCapture(args.input)
        if not self.cap.isOpened():
            print("Error: Couldn't open video source.")
            sys.exit()

        # Application UI layout
        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Debug
        self.label.setMouseTracking(True)
        self.setMouseTracking(True)

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
            keypoints_data = analyze_frame(frame, debug_window=self.proc_wind)

            zone_collisions = []
            detection_collisions = []

            original_height, original_width = frame.shape[:2]
            scale_x = 960 / original_width
            scale_y = 540 / original_height

            # Compute collisions
            for zone_index, zone in enumerate(self.zones):
                for zone_box in zone["boxes"]:
                    (zx1, zy1), (zx2, zy2) = zone_box

                    for label, (box, score) in keypoints_data.items():
                        x1, y1, x2, y2 = map(int, box)

                        # Scale coordinates
                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)

                        if not (scaled_x2 < zx1 or scaled_x1 > zx2 or scaled_y2 < zy1 or scaled_y1 > zy2):
                            zone_collisions.append(zone["name"])
                            detection_collisions.append(label)

            # Draw detections
            for label, (box, score) in keypoints_data.items():
                color = (0, 255, 0) if label not in detection_collisions else (0, 0, 255)
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            scaled_frame = cv2.resize(frame, (960, 540))

            # Draw zones
            for zone in self.zones:
                color = (0, 255, 0) if zone["name"] not in zone_collisions else (0, 0, 255)

                for box in zone["boxes"]:
                    (x1, y1), (x2, y2) = box
                    cv2.rectangle(scaled_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(scaled_frame, f"{zone['name']}", (x1 + 10, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Convert the scaled frame for display and output
            self.out_writer.write(scaled_frame)

            # Prepare to display the scaled image
            scaled_frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = scaled_frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(scaled_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))

            # Frame timing for consistent FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed_time)
            time.sleep(sleep_time)

        else:
            self.cap.release()
            self.out_writer.release()
            self.timer.stop()
            print(f"Video saved to {self.output_path}")
