import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt

#########################
# 新增：摄像头投影与反投影函数
#########################


#############################
# 1) 摄像头内参（示例）
#############################
f = 1020.0  # 焦距，像素
cx = 120.0
cy = 240.0
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]], dtype=np.float32)

#############################
# 全局变量：保存多帧候选 (E_i, n_i, P_i) 的列表
#############################
candidate_list = []
window_size = 20  # 可以根据实际情况调整

def ls_combine(candidates):
    """
    多帧最小二乘组合：
    对每个候选 (E_i, n_i)，计算矩阵 I - n_i n_i^T，
    然后求解：
      E = (sum_i (I - n_i n_i^T))^{-1} (sum_i (I - n_i n_i^T) E_i)
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I_mat = np.eye(3)
    for (Ei, ni, Pi) in candidates:
        I_minus = I_mat - np.outer(ni, ni)
        A += I_minus
        b += I_minus @ Ei
    if np.linalg.det(A) < 1e-6:
        return candidates[-1][0]  # 如果 A 奇异，则返回最后一个候选的 E_i
    E_final = np.linalg.inv(A) @ b
    return E_final


def backproject_point(p, K):
    """
    将图像平面中的点 p 反投影为归一化的 3D 射线
    p: (x,y) 像素坐标
    """
    p_h = np.array([p[0], p[1], 1.0], dtype=np.float32)
    invK = np.linalg.inv(K)
    ray = invK @ p_h
    ray = ray / np.linalg.norm(ray)
    return ray

def project_point(X3, K):
    """将 3D 点投影到 2D 像素坐标 (pinhole)"""
    # X3 = (X, Y, Z)
    x = (K[0,0]*X3[0]/X3[2] + K[0,2],
         K[1,1]*X3[1]/X3[2] + K[1,2])
    return (int(x[0]), int(x[1]))

def project_point(X, K):
    """
    将 3D 点 X 投影到图像平面（使用 pinhole 模型）
    X: np.array([X, Y, Z])
    """
    x = (K[0,0] * X[0] / X[2] + K[0,2], K[1,1] * X[1] / X[2] + K[1,2])
    return (int(x[0]), int(x[1]))

# #############################
# # 2) 3D 模型拟合简化
# #############################

# def fit_eye_model(pupil_ellipse, K, d=50.0, delta=40.0, eyeball_radius=12.0):
#     """
#     根据 2D 瞳孔椭圆拟合结果估计 3D 眼球模型参数
#       - pupil_ellipse: 由 cv2.fitEllipse 得到的结果 ((cx,cy), (axis1, axis2), angle)
#       - d: 假定瞳孔中心距离摄像头的距离（单位 mm）
#       - delta: 瞳孔中心到眼球中心的距离（单位 mm）
#       - eyeball_radius: 眼球半径（单位 mm）
#     返回：
#       E: 眼球中心（3D 坐标）
#       P3: 估计的 3D 瞳孔中心
#       gaze_vector: 视线向量（单位向量，从眼球中心指向瞳孔中心）
#     """
#     # 1. 从椭圆中心反投影，得到视线方向
#     center_2d = pupil_ellipse[0]
#     ray = backproject_point(center_2d, K)
    
#     # 2. 设定瞳孔中心到摄像头距离 d，得 P3
#     P3 = ray * d
    
#     # 3. 设置眼球中心为图像中心点对应方向上、距离为 d+R 的点
#     eyeball_ray = backproject_point((K[0,2], K[1,2]), K)
#     E = eyeball_ray * (d + eyeball_radius)
    
#     # 4. gaze vector
#     gaze_vector = (P3 - E) / np.linalg.norm(P3 - E)
    
#     return E, P3, gaze_vector


#############################
# 2) 3D 模型拟合 —— 从完整的2D椭圆重建3D圆
#############################
def fit_eye_model(pupil_ellipse, K, d=50.0, R=12.0):
    """
    基于完整的2D椭圆重建3D圆的方法：
    
    输入：
      - pupil_ellipse: ((c_x, c_y), (w, h), phi)
          其中 (c_x, c_y) 为椭圆中心，w, h 为椭圆宽、高，phi 为椭圆旋转角（单位度）
      - d: 假定瞳孔中心距离摄像头的深度（单位 mm），例如 50mm
      - R: 眼球中心到瞳孔中心的固定距离（单位 mm），例如 12mm
    
    算法步骤：
      1. 从2D椭圆中心反投影，得到摄像头坐标系中的单位方向 ray；
      2. 假设瞳孔中心位于 ray 上距离摄像头 d 处，得到 3D 瞳孔候选点 P = ray * d；
      3. 计算椭圆半轴： a = max(w,h)/2,  b = min(w,h)/2，并估计 tilt 角 theta = arccos(b/a);
      4. 将椭圆旋转角 phi 转换为弧度，并计算图像平面方向向量 v = [cos(phi), sin(phi), 0]；
      5. 构造圆平面的法向量： n = normalize([0,0,1]*cos(theta) + v*sin(theta));
      6. 根据约束，眼球中心 E = P - R * n；
      7. 视线向量 gaze = (P - E) / ||P - E||.
    
    返回：
      - E: 眼球中心的3D坐标
      - P: 估计的3D瞳孔中心
      - n: 圆平面法向量（可用于后续 disambiguation）
      - gaze_vector: 视线向量
    """
    # 1. 2D椭圆中心
    (cx, cy) = pupil_ellipse[0]
    # 反投影：得到单位方向 ray
    ray = backproject_point((cx, cy), K)
    
    # 2. 瞳孔中心候选点 P
    P = ray * d
    
    # 3. 计算半轴和 tilt 角 theta（防止除零）
    (w, h) = pupil_ellipse[1]
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0
    if a > 0:
        theta = math.acos(b / a)
    else:
        theta = 0.0
    
    # 4. 将椭圆旋转角转换为弧度，计算图像平面方向向量 v
    phi = math.radians(pupil_ellipse[2])
    v = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=np.float32)
    
    # 5. 构造法向量 n：以光轴 [0,0,1] 为基准
    optical_axis = np.array([0, 0, 1], dtype=np.float32)
    n = optical_axis * math.cos(theta) + v * math.sin(theta)
    n = n / np.linalg.norm(n)
    
    # 6. 估计眼球中心：E = P - R * n
    E = P - n * R
    
    # 7. 视线向量
    gaze_vector = (P - E) / np.linalg.norm(P - E)
    
    return E, P, n, gaze_vector


#############################
# 3) 在图像上可视化 —— 绘制出类似论文图中的效果：
#    1. 瞳孔椭圆（黄色）
#    2. 眼球大圆（蓝色），以眼球中心为圆心
#    3. 从眼球中心指向瞳孔中心的射线（绿色）
#    4. 用红色点标注眼球中心，蓝色点标注瞳孔中心
#############################
def draw_eye_model(result_frame, pupil_ellipse, E, P, eyeball_radius=12.0):
    """
    绘制效果：
      - 黄色椭圆：拟合的瞳孔轮廓
      - 蓝色大圆：眼球在图像中的投影（以 E 为圆心，半径由透视公式计算）
      - 绿色射线：从眼球中心 E 指向瞳孔中心 P
      - 红色点：眼球中心 E
      - 蓝色点：瞳孔中心 P
    """
    # 画瞳孔椭圆（黄色）
    cv2.ellipse(result_frame, pupil_ellipse, (0, 255, 255), 2)
    
    # 眼球中心投影
    eyeball_center_2d = project_point(E, K)
    # 根据透视公式计算眼球大圆在图像中的半径： r_proj = f * (eyeball_radius / E_z)
    if E[2] > 1e-3:
        r_proj = int(f * eyeball_radius / E[2])
    else:
        r_proj = 1
    # 画蓝色大圆（眼球轮廓）
    cv2.circle(result_frame, eyeball_center_2d, r_proj, (255, 0, 0), 2)
    
    # 用红色小圆标记眼球中心
    cv2.circle(result_frame, eyeball_center_2d, 3, (0, 0, 255), -1)
    
    # 瞳孔中心投影
    pupil_center_2d = project_point(P, K)
    # 用蓝色小圆标记瞳孔中心
    cv2.circle(result_frame, pupil_center_2d, 3, (255, 0, 0), -1)
    
    # 画从眼球中心到瞳孔中心的射线（绿色）
    cv2.line(result_frame, eyeball_center_2d, pupil_center_2d, (0, 255, 0), 2)
    
    return result_frame


# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=640, height=480):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image):

    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 5 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block

    return darkest_point

# New function to detect glint (brightest spot)
def get_brightest_area(image, ignoreBounds=20, blockSize=20, skip=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_sum = -1
    brightest_point = (0, 0)
    
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, skip):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, skip):
            block = gray[y:y+blockSize, x:x+blockSize]
            block_sum = np.sum(block)
            if block_sum > max_sum:
                max_sum = block_sum
                brightest_point = (x + blockSize//2, y + blockSize//2)
    return brightest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
  
    pupil_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            #gray_copy = gray_frame.copy()
            #cv2.drawContours(gray_copies[i-1], reduced_contours, -1, (255), 1)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        #cv2.drawContours(test_frame, final_contours, -1, (255, 255, 255), 1)
        ellipse = cv2.fitEllipse(final_contours[0])
        pupil_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        #cv2.circle(test_frame, darkest_point, 3, (255, 125, 125), -1)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #space
        cv2.putText(test_frame, "Q      = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #quit
        cv2.putText(test_frame, "D      = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #debug

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    #process_frames now returns a rotated rectangle for the ellipse for easy access
    return pupil_rotated_rect

#############################
# 修改 process_frame() —— 多帧最小二乘组合
#############################
# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):
    global candidate_list

    # 裁剪并缩放图像
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    brightest_point = get_brightest_area(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # 不同阈值下的二值化并限制ROI
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)
    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

    # 使用现有方法提取瞳孔椭圆（pupil_ellipse）
    pupil_ellipse = process_frames(
        thresholded_image_strict, thresholded_image_medium,
        thresholded_image_relaxed, frame, gray_frame, darkest_point,
        debug_mode_on=False, render_cv_window=False
    )

    # # 调用原有的 process_frames 处理得到瞳孔拟合椭圆
    # pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed,
    #                                      frame, gray_frame, darkest_point, False, False)

    result_frame = frame.copy()
    # 如果检测到有效椭圆（非全0）
    if pupil_ellipse[1] != (0, 0):
        # 计算单帧3D模型候选
        E_i, P_i, n_i, gaze_vec = fit_eye_model(pupil_ellipse, K, d=50.0, R=12.0)
        # 保存候选
        candidate_list.append((E_i, n_i, P_i))
        # 当候选达到一定数量后，进行最小二乘组合
        if len(candidate_list) >= window_size:
            E_global = ls_combine(candidate_list)
            # 可选：这里可以使用当前帧的 P_i 或对所有 P_i 求均值作为瞳孔中心参考
            # 这里取当前帧的 P_i
            gaze_vector_final = (P_i - E_global) / np.linalg.norm(P_i - E_global)
            # 在当前帧上显示多帧融合后的眼球中心
            result_frame = draw_eye_model(result_frame, pupil_ellipse, E_global, P_i, eyeball_radius=12.0)
            # 为了实现滑动窗口，可以选择保留最新的一部分候选：
            candidate_list = candidate_list[-window_size:]
        else:
            # 如果候选数未达到窗口大小，仍用当前帧的单帧估计进行显示
            result_frame = draw_eye_model(result_frame, pupil_ellipse, E_i, P_i, eyeball_radius=12.0)
    
    # 绘制 glint 点（黄色）作为参考
    cv2.circle(result_frame, brightest_point, 3, (0, 255, 255), -1)

    # pupil_center = tuple(map(int, pupil_rotated_rect[0]))  # (x, y)
    # cv2.circle(result_frame, pupil_center, 3, (255, 0, 0), -1)          # Blue dot: pupil
    # cv2.circle(result_frame, brightest_point, 3, (0, 255, 255), -1)     # Yellow dot: glint
    # print("Pupil center:", pupil_center)
    # print("Glint center:", brightest_point)
    # print("Image shape:", result_frame.shape)
    
    # # 如果拟合得到有效的椭圆，则计算 3D 眼模型
    # # 注意：当椭圆为 ( (0,0), (0,0), 0 ) 时，表示未检测到有效瞳孔
    # if pupil_rotated_rect[1] != (0, 0) and np.any(np.array(pupil_rotated_rect[0]) != 0):
    #     # 3D 眼模型拟合
    #     E, P3, gaze_vector = fit_eye_model(pupil_rotated_rect, K)
    #     print("Estimated 3D Eyeball Center (mm):", E)
    #     print("Estimated 3D Pupil Center (mm):", P3)
    #     print("Estimated Gaze Vector:", gaze_vector)
    #     # 在图像上绘制眼球模型及视线向量
    #     result_frame = draw_eye_model(result_frame, E, gaze_vector, K)

    return pupil_ellipse, brightest_point, result_frame

    # return pupil_rotated_rect, brightest_point, result_frame

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    # out = cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', fourcc, 30.0, (640, 480))  # Output video filename, codec, frame rate, and frame size
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))
    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(00, cv2.CAP_DSHOW)  # Camera input
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    debug_mode_on = False
    
    # temp_center = (0,0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用 process_frame() 处理当前帧
        pupil_ellipse, glint_point, result_frame = process_frame(frame)

        # 可选：打印坐标调试
        print("Pupil center:", pupil_ellipse[0])
        print("Glint center:", glint_point)

        # 显示带有 2D 瞳孔、3D 眼模型与视线向量的图像
        cv2.imshow('Pupil and 3D Eye Model Fitting', result_frame)
        # cv2.imshow('Pupil and Glint Tracking', result_frame)
        
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
    video_path = os.path.join(script_dir, "eye_test.mp4")
    
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


