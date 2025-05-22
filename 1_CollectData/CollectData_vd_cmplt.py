import sys
import cv2
import os
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget, QVBoxLayout, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor

class CalibrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Calibration")
        self.showMaximized()
        self.setStyleSheet("background-color: black;")

        # self.points = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
        #                (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
        #                (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
        self.current_index = -1
        screen = QDesktopWidget().screenGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.timer = QTimer()
        self.timer.timeout.connect(self.blink)
        self.blink_on = True
        self.blink_radius = 20  # 初始半径

        self.capture = cv2.VideoCapture(0)
        # Setup mode selector and output directory with mode in name
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["5p", "9p", "16p"])
        self.mode_selector.setStyleSheet("font-size: 24px;")
        self.mode_selector.setFixedWidth(160)
        selected_mode = self.mode_selector.currentText() if self.mode_selector.currentIndex() != -1 else "9p"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"eye_dataset_qt/session_{timestamp}_{selected_mode}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.labels = []

        self.button = QPushButton("开始采集", self)
        self.button.setStyleSheet("font-size: 28px;")
        self.button.clicked.connect(self.start_collection)

        layout = QVBoxLayout()
        layout.insertWidget(1, self.mode_selector, alignment=Qt.AlignCenter)
        layout.addStretch()
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

        self.setFocusPolicy(Qt.StrongFocus)

        # Initialize video writer and path
        ret, frame = self.capture.read()
        if ret:
            h, w = frame.shape[:2]
            video_fname = f"eye_full_video.mp4"
            self.video_fpath = os.path.join(self.output_dir, video_fname)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.video_fpath, fourcc, 30, (w, h))
            self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write the first frame to start
            self.writer.write(frame)
        else:
            self.writer = None
            self.video_fpath = None
            self.start_time = None

        # Add a video timer to write frames at a constant frame rate
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.write_video_frame)

    def start_collection(self):
        mode = self.mode_selector.currentText()
        if mode == "5p":
            self.points = [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        elif mode == "16p":
            self.points = [(x, y) for y in [0.2, 0.4, 0.6, 0.8] for x in [0.2, 0.4, 0.6, 0.8]]
        else:  # 默认9点
            self.points = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
                        (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
                        (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
            
        self.button.hide()
        self.video_timer.start(33)  # roughly 30 fps
        self.next_point()

    def next_point(self):
        self.current_index += 1
        if self.current_index >= len(self.points):
            if self.writer is not None:
                self.writer.release()
            if self.capture.isOpened():
                self.capture.release()
            self.video_timer.stop()
            # Save labels with gaze points and timestamps
            with open(os.path.join(self.output_dir, "labels.json"), "w") as f:
                json.dump(self.labels, f, indent=2)
            self.close()
            return
        self.timer.start(40)  # 控制红点动画刷新频率（更快，平滑动画）
        self.blink_anim_phase = 0  # 用于动画进度
        self.blink_anim_direction = -1  # 1为增大，-1为减小
        self.blink_radius = 20  # 重置半径
        self.point_start_time = datetime.now()  # 记录采集点开始时间
        self.point_start_time_str = self.point_start_time.strftime("%Y-%m-%d %H:%M:%S")
        QTimer.singleShot(3000, self.capture_frames)

    def capture_frames(self):
        # For this method, just capture the gaze point and timestamp
        x_norm, y_norm = self.points[self.current_index]
        x = int(x_norm * self.screen_w)
        y = int(y_norm * self.screen_h)
        # Append current gaze point with timestamp and duration
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_sec = (datetime.now() - self.point_start_time).total_seconds() if hasattr(self, "point_start_time") else None
        self.labels.append({
            "point_index": self.current_index,
            "gaze_xy": [x, y],
            "start_timestamp": self.point_start_time_str,
            "end_timestamp": current_time,
            "duration_sec": duration_sec
        })
        self.next_point()

    def write_video_frame(self):
        if self.writer is not None:
            ret, frame = self.capture.read()
            if ret:
                self.writer.write(frame)

    def blink(self):
        # 红点大小来回变化动画
        # 半径在20~40之间变化，周期约1.2秒（对应timer 40ms, 30步）
        min_radius = 4
        max_radius = 20
        steps = 30
        if not hasattr(self, "blink_anim_phase"):
            self.blink_anim_phase = 0
            self.blink_anim_direction = 1
        # 在0~steps之间往返
        self.blink_anim_phase += self.blink_anim_direction
        if self.blink_anim_phase >= steps:
            self.blink_anim_phase = steps
            self.blink_anim_direction = -1
        elif self.blink_anim_phase <= 0:
            self.blink_anim_phase = 0
            self.blink_anim_direction = 1
        # 线性插值半径
        t = self.blink_anim_phase / steps
        self.blink_radius = int(min_radius + (max_radius - min_radius) * t)
        self.blink_on = True
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor(0, 0, 0))
        if self.current_index >= 0 and self.current_index < len(self.points):
            qp.setRenderHint(QPainter.Antialiasing)
            if self.blink_on:
                qp.setBrush(QColor(255, 0, 0))
                x = int(self.points[self.current_index][0] * self.screen_w)
                y = int(self.points[self.current_index][1] * self.screen_h)
                r = self.blink_radius
                qp.drawEllipse(x - r, y - r, 2 * r, 2 * r)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.writer is not None:
                self.writer.release()
            if self.capture.isOpened():
                self.capture.release()
            self.video_timer.stop()
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalibrationApp()
    win.show()
    sys.exit(app.exec_())