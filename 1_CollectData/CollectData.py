import sys
import cv2
import os
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor

class CalibrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Calibration")
        self.showMaximized()
        self.setStyleSheet("background-color: black;")
        self.points = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
                       (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
                       (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
        self.current_index = -1
        screen = QDesktopWidget().screenGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.timer = QTimer()
        self.timer.timeout.connect(self.blink)
        self.blink_on = True

        self.capture = cv2.VideoCapture(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"eye_dataset_qt/session_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.labels = []

        self.button = QPushButton("开始采集", self)
        self.button.setStyleSheet("font-size: 28px;")
        self.button.clicked.connect(self.start_collection)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

        self.setFocusPolicy(Qt.StrongFocus)

    def start_collection(self):
        self.button.hide()
        self.next_point()

    def next_point(self):
        self.current_index += 1
        if self.current_index >= len(self.points):
            self.capture.release()
            with open(os.path.join(self.output_dir, "labels.json"), "w") as f:
                json.dump(self.labels, f, indent=2)
            self.close()
            return
        self.timer.start(400)  # 控制闪烁
        QTimer.singleShot(1300, self.capture_frames)

    def capture_frames(self):
        images = []
        x_norm, y_norm = self.points[self.current_index]
        x = int(x_norm * self.screen_w)
        y = int(y_norm * self.screen_h)
        for i in range(30):  # 30 帧
            ret, frame = self.capture.read()
            if ret:
                fname = f"eye_{self.current_index:02d}_{i:02d}.jpg"
                fpath = os.path.join(self.output_dir, fname)
                cv2.imwrite(fpath, frame)
                images.append(fname)
        self.labels.append({"point_index": self.current_index, "gaze_xy": [x, y], "images": images})
        self.next_point()

    def blink(self):
        self.blink_on = not self.blink_on
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
                qp.drawEllipse(x - 20, y - 20, 40, 40)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalibrationApp()
    win.show()
    sys.exit(app.exec_())