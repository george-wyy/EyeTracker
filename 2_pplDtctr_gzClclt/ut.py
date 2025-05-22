import cv2
import numpy as np

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

def find_glints_near_pupil(image, pupil_center, search_size=90):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y = int(pupil_center[0]), int(pupil_center[1])
    h, w = gray.shape

    # 限制搜索区域在暗瞳周围
    x1, y1 = max(0, x - search_size), max(0, y - search_size)
    x2, y2 = min(w, x + search_size), min(h, y + search_size)
    roi = gray[y1:y2, x1:x2]

    # 二值化高亮区域
    _, thresh = cv2.threshold(roi, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glints = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 <= area <= 80:  # 控制在 5–10px 大小范围
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                glints.append((cx, cy))

    # 返回最接近水平排列的两个 glint 点
    if len(glints) >= 2:
        glints.sort(key=lambda p: p[0])  # 按 x 排序
        min_pair = min(((p1, p2) for i, p1 in enumerate(glints) for p2 in glints[i+1:]),
                       key=lambda pair: abs(pair[0][1] - pair[1][1]))
        return min_pair
    elif len(glints) == 1:
        return (glints[0], glints[0])
    else:
        return ((0, 0), (0, 0))