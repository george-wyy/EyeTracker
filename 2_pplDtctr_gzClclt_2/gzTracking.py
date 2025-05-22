import sys
import os
import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QColor
from detectors import process_frame


# Gaze Tracking App
class GazeTrackingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Tracking")
        self.setGeometry(100, 100, 1920, 1080)
        self.model = joblib.load(os.path.join(os.path.dirname(__file__), "best_model.pkl"))
        print("Model expects", self.model.named_steps['standardscaler'].n_features_in_, "features")
        
        # self.use_camera = False  # 切换 True 为摄像头输入
        self.use_camera = True  # 切换 False 为离线视频输入
        if self.use_camera:
            self.cap = cv2.VideoCapture(0)
        else:
            video_path = os.path.join(os.path.dirname(__file__), "test_video.mp4")
            self.cap = cv2.VideoCapture(video_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)
        self.gaze_pos = None
        self.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        pupil_ellipse, glint1, glint2, _ = process_frame(frame)
        if glint1 == (0, 0) or glint2 == (0, 0):
            self.gaze_pos = None
            return
        pupil = pupil_ellipse[0]
        vec_dx = pupil[0] - (glint1[0] + glint2[0]) / 2
        vec_dy = pupil[1] - (glint1[1] + glint2[1]) / 2
        glint_dist = np.linalg.norm(np.array(glint1) - np.array(glint2))
        pupil_to_glint_dist = np.linalg.norm(np.array(pupil) - np.array([(glint1[0]+glint2[0])/2, (glint1[1]+glint2[1])/2]))
        
        X = np.array([[pupil[0], pupil[1],
               glint1[0], glint1[1],
               glint2[0], glint2[1],
               vec_dx, vec_dy,
               pupil_to_glint_dist]])
        self.gaze_pos = self.model.predict(X)[0]
        self.update()

        # 画 gaze 点在视频上并显示
        if self.gaze_pos is not None and all(np.isfinite(self.gaze_pos)):
            gx, gy = int(self.gaze_pos[0]), int(self.gaze_pos[1])
            if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
                cv2.circle(frame, (gx, gy), 10, (0, 255, 0), -1)
        cv2.imshow("Video with Gaze", frame)
        cv2.waitKey(1)

    def paintEvent(self, event):
        if self.gaze_pos is not None and all(np.isfinite(self.gaze_pos)):
            x, y = int(self.gaze_pos[0]), int(self.gaze_pos[1])
            if 0 <= x <= self.width() and 0 <= y <= self.height():
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setBrush(QColor(0, 255, 0, 180))
                painter.drawEllipse(x - 10, y - 10, 20, 20)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GazeTrackingApp()
    sys.exit(app.exec_())