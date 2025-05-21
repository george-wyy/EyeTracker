def enhance_contrast(gray):
    # 使用 CLAHE 进行自适应直方图均衡化以增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced
import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels

# Expand nested labels to a flat list: one dict per image/gaze point
def expand_labels(grouped_labels):
    flat_labels = []
    for group in grouped_labels:
        gaze_point = group['gaze_xy']
        for img_name in group['images']:
            flat_labels.append({
                'eye_image_path': img_name,
                'gaze_point': gaze_point
            })
    return flat_labels

def get_darkest_area(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (7,7), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_ellipse = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000 and len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            if area > max_area and y > 50:
                max_area = area
                best_ellipse = ellipse
    return best_ellipse

def apply_binary_threshold(gray):
    # Use median blur to reduce noise, then adaptive threshold for better handling of gray pupil images
    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )
    return binary

def mask_outside_square(binary_img, square_size=100):
    h, w = binary_img.shape
    mask = np.zeros_like(binary_img)
    center_x, center_y = w // 2, h // 2
    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
    masked = cv2.bitwise_and(binary_img, mask)
    return masked

def process_frames(gray_img, output_dir=None, idx=None):
    # Step 1: Median blur to reduce noise
    blurred = cv2.medianBlur(gray_img, 5)
    # Step 2: Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    # Step 3: Mask outside central square (focus on center)
    masked = mask_outside_square(binary, square_size=250)
    # Step 4: Find contours
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Save intermediate images if output_dir and idx are provided
    if output_dir is not None and idx is not None:
        binary_path = os.path.join(output_dir, f'binary_{idx}.png')
        masked_path = os.path.join(output_dir, f'masked_{idx}.png')
        contours_path = os.path.join(output_dir, f'contours_{idx}.png')
        cv2.imwrite(binary_path, binary)
        cv2.imwrite(masked_path, masked)
        # Draw all contours for visualization
        vis_contours = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_contours, contours, -1, (0,255,0), 2)
        cv2.imwrite(contours_path, vis_contours)

    best_score = -np.inf
    best_ellipse = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 10000 and len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            # Area ratio test: contour area / ellipse area
            ellipse_area = np.pi * (MA / 2) * (ma / 2)
            if ellipse_area <= 0:
                continue
            area_ratio = area / ellipse_area
            # Ellipse flatness (1 - min/max axis ratio, closer to 1 is more circular)
            flatness = min(MA, ma) / max(MA, ma)
            # Constraints: area ratio ~0.6-1.2, flatness > 0.5 (not too flat)
            if 0.6 < area_ratio < 1.2 and flatness > 0.5:
                # Score: prioritize area_ratio near 1 and flatness near 1
                score = -abs(area_ratio - 1) + flatness
                if score > best_score:
                    best_score = score
                    best_ellipse = ellipse
    if best_ellipse is None:
        return None, None, None
    (x, y), (MA, ma), angle = best_ellipse
    center = (int(x), int(y))
    radius = int(max(MA, ma) / 2)
    return center, radius, best_ellipse

def get_brightest_area(gray_img):
    _, thresh = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reflection_centers = []
    min_area = 3
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                reflection_centers.append(((cx, cy), area))
    reflection_centers.sort(key=lambda x: x[1], reverse=True)
    return reflection_centers[:2]

def extract_pupil_and_reflection(eye_img, output_dir, idx):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    # 图像增强：增强对比度
    enhanced_gray = enhance_contrast(gray)

    # Save enhanced image
    enhanced_path = os.path.join(output_dir, f"enhanced_{idx}.png")
    cv2.imwrite(enhanced_path, enhanced_gray)

    # Use integrated darkest area detection process, with debug saving of intermediate images
    pupil_center, pupil_radius, pupil_ellipse = process_frames(enhanced_gray, output_dir=output_dir, idx=idx)

    if pupil_ellipse is not None:
        cv2.ellipse(eye_img, pupil_ellipse, (0,255,255), 2)

    # Use get_brightest_area to detect reflections
    top_reflections = get_brightest_area(enhanced_gray)

    for (cx, cy), _ in top_reflections:
        cv2.circle(eye_img, (cx, cy), 3, (255, 0, 0), -1)  # 蓝点
        cv2.rectangle(eye_img, (cx-5, cy-5), (cx+5, cy+5), (0, 0, 255), 1)  # 红框

    reflection_centers = [pt for pt, _ in top_reflections]

    return pupil_center, pupil_radius, reflection_centers, eye_img

def main():
    from collections import defaultdict
    label_file = '/Users/yiyanwang/wkspaces/DIYEyeTracker/EyeTracker/eye_dataset_qt/session_20250520_205439/labels.json'
    session_name = os.path.basename(os.path.dirname(label_file))
    output_dir = os.path.join('processed_images', session_name)
    os.makedirs(output_dir, exist_ok=True)
    grouped_labels = load_labels(label_file)
    labels = expand_labels(grouped_labels)

    pupil_points = []
    reflection_points = []
    gaze_points = []

    success_count = defaultdict(int)
    fail_count = defaultdict(int)

    for idx, item in enumerate(labels):
        eye_img_path = os.path.join(os.path.dirname(label_file), item['eye_image_path'])
        gaze_point = item['gaze_point']  # [x, y]

        # Extract xx_yy from filename
        base_name = os.path.basename(eye_img_path)
        name_part = os.path.splitext(base_name)[0]
        parts = name_part.split('_')
        xx_yy = parts[1] if len(parts) > 1 else 'unknown'

        # Only process images with point number 00, 04, 08
        if xx_yy not in ['00',]:
            continue

        eye_img = cv2.imread(eye_img_path)
        if eye_img is None:
            print(f"Warning: Could not load image {eye_img_path}")
            fail_count[xx_yy] += 1
            continue

        pupil_center, pupil_radius, reflection_centers, vis_img = extract_pupil_and_reflection(eye_img, output_dir, idx)

        if pupil_center is None or len(reflection_centers) == 0:
            print(f"[FAIL] Could not detect pupil or reflection in image {eye_img_path}")
            fail_count[xx_yy] += 1
            continue

        pupil_points.append(pupil_center)
        # Use the first reflection center as representative
        reflection_points.append(reflection_centers[0])
        gaze_points.append(gaze_point)

        success_count[xx_yy] += 1

        # Draw session name at top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, session_name, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save visualization image
        save_path = os.path.join(output_dir, f'processed_{idx}.png')
        cv2.imwrite(save_path, vis_img)

    # Print success/failure counts per eye_xx
    for key in sorted(set(list(success_count.keys()) + list(fail_count.keys()))):
        print(f"eye_{key}: Successes = {success_count[key]}, Failures = {fail_count[key]}")

    # Check data completeness before training
    if len(pupil_points) == 0 or len(reflection_points) == 0 or len(gaze_points) == 0:
        print("Error: No valid data samples collected for training.")
        print(f"Collected samples - pupil_points: {len(pupil_points)}, reflection_points: {len(reflection_points)}, gaze_points: {len(gaze_points)}")
        return
    else:
        print(f"Collected {len(pupil_points)} valid samples for training.")

    # Prepare data for regression
    pupil_points = np.array(pupil_points)
    reflection_points = np.array(reflection_points)
    gaze_points = np.array(gaze_points)

    # Use difference between pupil and reflection as features
    features = pupil_points - reflection_points

    poly = PolynomialFeatures(degree=2)
    features_poly = poly.fit_transform(features)

    model = LinearRegression()
    model.fit(features_poly, gaze_points)

    gaze_pred = model.predict(features_poly)

    # Save model coefficients
    np.save('gaze_model_coef.npy', model.coef_)
    np.save('gaze_model_intercept.npy', model.intercept_)

    # Plot results
    plt.figure()
    plt.scatter(gaze_points[:,0], gaze_points[:,1], label='True Gaze', c='blue')
    plt.scatter(gaze_pred[:,0], gaze_pred[:,1], label='Predicted Gaze', c='red', marker='x')
    plt.legend()
    plt.title('Gaze Mapping: True vs Predicted')
    plt.xlabel('Gaze X')
    plt.ylabel('Gaze Y')
    plt.savefig('gaze_mapping_result.png')
    plt.show()

if __name__ == '__main__':
    main()
