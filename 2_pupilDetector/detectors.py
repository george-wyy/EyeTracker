import cv2
import numpy as np

import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from  ut import crop_to_aspect_ratio, apply_binary_threshold, mask_outside_square, get_darkest_area, get_brightest_area, find_glints_near_pupil

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
  
    final_rotated_rect = ((0,0),(0,0),0)

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
        final_rotated_rect = ellipse
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
    return final_rotated_rect


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)
    # Also find the brightest point for glint
    # brightest_point = get_brightest_area(frame)
    

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
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)

    # Optional: draw both points on the frame
    result_frame = frame.copy()
    pupil_center = tuple(map(int, final_rotated_rect[0]))  # (x, y)

    glint1, glint2 = find_glints_near_pupil(frame, pupil_center)
    
    cv2.circle(result_frame, pupil_center, 3, (0, 255, 0), -1)          # Blue dot: pupil
    # cv2.circle(result_frame, brightest_point, 3, (0, 0, 255), -1)     # red dot: glint
    cv2.circle(result_frame, glint1, 3, (0, 0, 255), -1)     # red dot: glint
    cv2.circle(result_frame, glint2, 3, (0, 0, 255), -1)     # red dot: glint
    print("Pupil center:", pupil_center)
    print("Glint center:", glint2)
    print("Image shape:", result_frame.shape)
    
    return final_rotated_rect, glint2, result_frame
