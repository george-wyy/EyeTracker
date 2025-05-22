import cv2
import sys
import os
# # 将 2_pupilDetector 路径添加到 sys.path 中
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)

import tkinter as tk
from tkinter import filedialog

from detectors import process_frame, process_frames
from ut import crop_to_aspect_ratio, get_darkest_area, apply_binary_threshold, mask_outside_square

# 添加标签
import json
from datetime import datetime

def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [{
        'index': d['point_index'],
        'gaze_xy': d['gaze_xy'],
        'timestamp': datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
    } for d in data]

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', fourcc, 30.0, (640, 480))  # Output video filename, codec, frame rate, and frame size

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)

        from datetime import timedelta
        labels = load_labels(os.path.join(os.path.dirname(video_path), "labels.json"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = labels[0]['timestamp'] - timedelta(seconds=1)
    elif input_method == 2:
        cap = cv2.VideoCapture(00, cv2.CAP_DSHOW)  # Camera input 这里可以相机实时输入
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    debug_mode_on = False
    
    temp_center = (0,0)

    frame_idx = 0
    collected_pupils = []
    collected_gazes = []
    collected_raw = []
    while True:
        ret, frame = cap.read()
        
        current_time = start_time + timedelta(seconds=frame_idx / fps)
        frame_idx += 1

        if not ret:
            break

        # 用 process_frame() 处理当前帧
        pupil_ellipse, glint_point1, glint_point2, result_frame = process_frame(frame)

        pupil_center = pupil_ellipse[0]

        # for label in labels:
        #     if abs((current_time - label['timestamp']).total_seconds()) < 0.5:
        #         if 'pupil_samples' not in label:
        #             label['pupil_samples'] = []
        #         label['pupil_samples'].append(pupil_center)



        for label in labels:
            if abs((current_time - label['timestamp']).total_seconds()) < 0.3:
                collected_pupils.append(pupil_center)
                collected_gazes.append(label['gaze_xy'])
                collected_raw.append({
                    'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    'gaze_x': label['gaze_xy'][0],
                    'gaze_y': label['gaze_xy'][1],
                    'pupil_x': pupil_center[0],
                    'pupil_y': pupil_center[1],
                    'glint1_x': glint_point1[0],
                    'glint1_y': glint_point1[1],
                    'glint2_x': glint_point2[0],
                    'glint2_y': glint_point2[1]
                })

        # 可选：打印坐标调试
        print("Pupil center:", pupil_ellipse[0])
        # print("Glint center:", glint_point)
        


        # 显示带标注的图像
        cv2.imshow('Pupil and Glint Tracking', result_frame)
        
        # 写入视频
        out.write(result_frame)

        # Crop and resize frame
        frame = crop_to_aspect_ratio(frame)

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
        
        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
        
        #take the three images thresholded at different levels and process them
        pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, True)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d') and debug_mode_on == False:  # Press 'q' to start debug mode
            debug_mode_on = True
        elif key == ord('d') and debug_mode_on == True:
            debug_mode_on = False
            cv2.destroyAllWindows()
        if key == ord('q'):  # Press 'q' to quit
            out.release()
            break   
        elif key == ord(' '):  # Press spacebar to start/stop
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Press spacebar again to resume
                    break
                elif key == ord('q'):  # Press 'q' to quit
                    break

    cap.release()
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # 拟合 gaze 映射模型（基于所有帧样本）
    eye_features = np.array(collected_pupils)
    gaze_targets = np.array(collected_gazes)

    if len(eye_features) >= 2:
        model = LinearRegression().fit(eye_features, gaze_targets)
        preds = model.predict(eye_features)
        df = pd.DataFrame({
            'gaze_x': gaze_targets[:,0],
            'gaze_y': gaze_targets[:,1],
            'pupil_x': eye_features[:,0],
            'pupil_y': eye_features[:,1],
            'pred_x': preds[:,0],
            'pred_y': preds[:,1]
        })
        csv_path = os.path.join(os.path.dirname(video_path), "calibration_result.csv")
        df.to_csv(csv_path, index=False)
        print(f"Calibration results saved to {csv_path}")

    if collected_raw:
        raw_df = pd.DataFrame(collected_raw)
        raw_csv_path = os.path.join(os.path.dirname(video_path), "raw_eye_data.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Raw eye data saved to {raw_csv_path}")

    out.release()
    cv2.destroyAllWindows()

#Prompts the user to select a video file if the hardcoded path is not found
#This is just for my debugging convenience :)
def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    # video_path = 'C:/Storage/Google Drive/Eye Tracking/fulleyetest3.mp4'
    # 获取当前.py文件所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # video_path = os.path.join(script_dir, "eye_dataset_qt/session_20250520_230607/eye_full_video.mp4")
    video_path = os.path.join(script_dir, "..", "eye_dataset_qt", "session_20250520_230541", "eye_full_video.mp4")
    
    if not os.path.exists(video_path):
        print("No file found at hardcoded path. Please select a video file.")
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
        if not video_path:
            print("No file selected. Exiting.")
            return
            
    #second parameter is 1 for video 2 for webcam
    process_video(video_path, 1)

if __name__ == "__main__":
    select_video()


