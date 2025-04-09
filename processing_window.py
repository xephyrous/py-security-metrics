import json
import os.path

import cv2
import sys
import time

import numpy
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QMenu, QMenuBar, QSizePolicy

from models.yolo11 import analyze_frame, draw_detections
from ui.smart_label import SmartLabel


class ProcessingWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.menu_bar = None
        self.cursor_timer = None
        self.pos_menu = None
        self.video_label = None
        self.proc_wind = None

        if args is not None:
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
        self.initialize_ui(args is not None)

        # FPS
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.output_path = None
        self.out_writer = None

        # Begin processing if invoked through CLI
        if args is not None:
            # Frame data
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.output_path = args.output
            self.out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

            self.start_processing_timer()

    def update_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()

        if ret:
            keypoints_data = analyze_frame(frame, debug_window=self.proc_wind)

            zone_collisions = []
            detection_collisions = []

            original_height, original_width = frame.shape[:2]
            scale_x = original_width / 960
            scale_y = original_height / 540

            # Compute collisions
            for zone_index, zone in enumerate(self.zones):
                for zone_box in zone["boxes"]:
                    (zx1, zy1), (zx2, zy2) = zone_box

                    scaled_zx1 = int(zx1 * scale_x)
                    scaled_zy1 = int(zy1 * scale_y)
                    scaled_zx2 = int(zx2 * scale_x)
                    scaled_zy2 = int(zy2 * scale_y)

                    for label, (box, score, feet) in keypoints_data.items():
                        x1, y1, x2, y2 = map(int, box)

                        if not (x2 < scaled_zx1 or x1 > scaled_zx2 or y2 < scaled_zy1 or y1 > scaled_zy2):
                            zone_collisions.append(zone["name"])
                            detection_collisions.append(label)

            # Draw detections
            for label, (box, score, feet) in keypoints_data.items():
                color = (0, 255, 0) if label not in detection_collisions else (0, 0, 255)
                x1, y1, x2, y2 = map(int, box)
                xr1, yr1, xr2, yr2 = map(int, feet)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (xr1, yr1), (xr2, yr2), color, 2)
                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw zones
            for zone in self.zones:
                color = (0, 255, 0) if zone["name"] not in zone_collisions else (0, 0, 255)

                for box in zone["boxes"]:
                    (x1, y1), (x2, y2) = box

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{zone['name']}", (x1 + 10, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Write frame to output video
            self.out_writer.write(frame)

            # Display frame to UI
            win_size = self.size()
            scaled_frame = cv2.resize(frame, (win_size.width(), win_size.height()))
            scaled_frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = scaled_frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(scaled_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

            # Frame timing for consistent FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, int(self.frame_time - elapsed_time))
            time.sleep(sleep_time)

        else:
            self.cap.release()
            self.out_writer.release()
            self.timer.stop()
            print(f"Video saved to {self.output_path}")

    def start_processing_timer(self):
        self.timer.start(int(self.frame_time * 1000))

    def initialize_ui(self, is_cli):
        self.video_label = SmartLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if is_cli:
            layout.addWidget(self.video_label)
        else:
            # Base menus
            self.menu_bar = self.menuBar()
            file_menu = self.menu_bar.addMenu("File")
            edit = self.menu_bar.addMenu("Edit")
            view = self.menu_bar.addMenu("View")

            # Cursor position
            pos_bar = QMenuBar(self.menu_bar)
            self.pos_menu = QMenu("[0, 0]", pos_bar)
            pos_bar.addMenu(self.pos_menu)
            self.menu_bar.setCornerWidget(pos_bar)

            # Cursor position updating
            self.cursor_timer = QTimer(self)
            self.cursor_timer.timeout.connect(self.update_cursor_pos)
            self.cursor_timer.start(100)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def update_cursor_pos(self):
        pos = self.video_label.mouse_position
        self.pos_menu.setTitle(f"[{pos[0]}, {pos[1]}]")