# Building a lane perception program that can take road images and dashcam video as input and highlight lanes from it. 

from cgitb import grey
from doctest import debug
from PIL.ImageQt import rgb
import cv2
import numpy as np  
import matplotlib.pyplot as plt 

# Function to create a region on interest, which in our case is a triangle. This helps us 
# Mask the other parts of the image like trees, fences, branches, etc. 

def region_of_interest (image): 
    # define height and width of grayscale image
    img_height = image.shape[0]
    img_width = image.shape[1]
    horizon_y = int(img_height * 0.2)

    # Define our triangle polygon (Bottom-Left, Top-Peak, Bottom-Right)
    img_triangle = np.array([[(0, img_height), (img_width // 2, horizon_y), (img_width, img_height)]])

    # Create a black mask of the given width and height
    black_mask = np.zeros_like(image)

    # Fill the triangle with white
    cv2.fillPoly(black_mask, img_triangle, 255)

    masked_image = cv2.bitwise_and(image, black_mask)
    return masked_image

# Helper function to make co-ordinates    
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * (3/5)) # Slightly lower than the middle
    
    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

# Helper function to draw lines from the co-ordinates we plant to get from cv2.HoughLinesP
def draw_lines(image, lines): 
    # Create black image of the image with same dimensions
    line_image = np.zeros_like(image)

    left_slopes = []
    right_slopes = []

    if lines is not None: 
        for line in lines:
            # Get co-ordinates
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0: 
                slope_m = 0
            else:
                slope_m = (y2 - y1)/(x2 - x1)
            intercept = y1 - slope_m*x1

            if abs(slope_m) > 0.5:
                if slope_m <= 0: 
                    left_slopes.append([slope_m, intercept])
                else: 
                    right_slopes.append([slope_m, intercept])

    if len(left_slopes) > 0:
          avg_left_slope = np.average(left_slopes, axis=0)
    else:
        avg_left_slope = None
    
    if len(right_slopes) > 0:
          avg_right_slope = np.average(right_slopes, axis=0)
    else:
        avg_right_slope = None
    

    # Calculate coordinates for Left Lane
    if avg_left_slope is not None: # Check if we actually found a line
        left_line = make_coordinates(image, avg_left_slope)
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Calculate coordinates for Right Lane
    if avg_right_slope is not None:
        right_line = make_coordinates(image, avg_right_slope)
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

# Importing test_lane image and converting to grayscale
img_test_lane = cv2.imread('./Images/test_lane.jpg')
img_test_lane_rgb = cv2.cvtColor(img_test_lane, cv2.COLOR_BGR2RGB)

# Check if image is loaded or not
if img_test_lane is None: 
    print("Error - no image found")

else: # Convert to grayscale
    img_test_lane_gray = cv2.cvtColor(img_test_lane, cv2.COLOR_BGR2GRAY)

#  Next we need to apply a gaussian blur in order to smoothen the image, this is to smoothen the images so that a computer doesn't
#  Mistake gravel or coarse edges for lines. 

img_test_lane_blur = cv2.GaussianBlur(img_test_lane_gray, (5, 5), 0)
img_test_lane_median_blur = cv2.medianBlur(img_test_lane_gray, 5, 0)

# Now to find edge detection - for edge detection, we rely on the rate of change or gradient. For example, 
# if there is a sharp change between gravel and white line, there's a gradient there. We use gradients for 
# Edge detection

# Canny Algorithm/Tool 50, 150 are the thresholds we're using, others for experiementation.

img_test_lane_canny1 = cv2.Canny(img_test_lane_blur, 50, 150)
img_test_lane_canny2 = cv2.Canny(img_test_lane_blur, 40, 160)
img_test_lane_canny3 = cv2.Canny(img_test_lane_blur, 60, 140)
img_test_lane_cropped = region_of_interest(cv2.Canny(img_test_lane_median_blur, 50, 150))

# Hough Transform to find lines
# rho=2, theta=1 degree, threshold=100 votes
# minLineLength=40 pixels, maxLineGap=5 pixels (connects dashed lines!)
lines = cv2.HoughLinesP(img_test_lane_cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Draw lines on black image
black_line_image = draw_lines(img_test_lane, lines)

# Showing them together to highlight the difference (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()  # 1D array so we can index 0..5

# Display the images in the respective subplots
images = [
    (img_test_lane_rgb, 'Original'),
    (img_test_lane_gray, 'Grayscale'),
    (img_test_lane_blur, 'Gaussian blur'),
    (img_test_lane_median_blur, 'Median Blur'),
    (img_test_lane_canny1, 'Canny (50, 150)'),
    (img_test_lane_cropped, 'Canny cropped (50, 150)')
]
for i, (img, title) in enumerate(images):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(title, fontsize=11)
    axes[i].axis('off')  # cleaner look: no ticks/axes

# Hide the empty 6th subplot
axes[5].axis('off')

fig.suptitle('Lane perception pipeline', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Blend the original image with the line image
# 0.8 = keep 80% of the original road brightness
# 1.0 = keep 100% of the blue line brightness
combo_image = cv2.addWeighted(img_test_lane, 0.8, black_line_image, 1, 1)

# Show the final result
# We convert BGR (OpenCV format) to RGB (Matplotlib format) so colors look right
plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
plt.show()

# Video Capture
video_lane = cv2.VideoCapture('./Videos/solidWhiteRight.mp4')

# Set up VideoWriter to save the processed video (same size and FPS as input)
width = int(video_lane.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_lane.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_lane.get(cv2.CAP_PROP_FPS)
output_path = './Videos/output_lanes.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while(video_lane.isOpened()):
   ret, frame = video_lane.read()

   if not ret: 
    break
   else:
    #cv2.imshow("Test frame from video", frame)
    canny_image = cv2.Canny(frame, 50, 150)
    cropped_canny_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    black_line_canny = draw_lines(frame, lines)
    combo_frame = cv2.addWeighted(frame, 0.8, black_line_canny, 1, 1)
    out.write(combo_frame)
    cv2.imshow("Test frame from vid", combo_frame)

   # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_lane.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved processed video to {output_path}")

# Creating a bird's eye view to mimic a drone view - this will help us handle cases where there are curves on the road. 
# Idea is to use the cropped triangle image and chop off a portion at the head. Then, we use the points of the remaining trapezoid
# to do a perspective transform using cv2.perspectiveTransform() to give us the new values. 

# Define src and dst for perspective transform lane
h, w = img_test_lane.shape[:2]

src = np.float32([
    [w * 0.025, h],       # Bottom-Left
    [w * 0.975, h],       # Bottom-Right
    [w * 0.70, h * 0.65], # Top-Right (Example ratio)
    [w * 0.30, h * 0.65]  # Top-Left (Example ratio)
])

dst = np.float32([
    [w * 0.20, h],       # Bottom-Left
    [w * 0.80, h],       # Bottom-Right
    [w * 0.80, 0],       # Top-Right
    [w * 0.20, 0]        # Top-Left
])

print("Image shape:", img_test_lane.shape)

# Chop off the image and lets see what it looks like
# Create a copy so we don't mess up the original
debug_image = img_test_lane_rgb.copy()

# Draw the 4 source points
for point in src:
    x, y = int(point[0]), int(point[1])
    cv2.circle(debug_image, (x, y), 10, (255, 0, 0), -1) # Red dots

# Draw the polygon connecting them
pts = src.reshape((-1, 1, 2)).astype(np.int32)
cv2.polylines(debug_image, [pts], True, (255, 0, 0), 3)

plt.imshow(debug_image)
plt.show()

M = cv2.getPerspectiveTransform(src, dst)

# We need the width and height of the image for the last argument
img_size = (img_test_lane_rgb.shape[1], img_test_lane_rgb.shape[0]) # (width, height)
warped_image = cv2.warpPerspective(img_test_lane_gray, M, img_size)
plt.imshow(warped_image, cmap='gray')
plt.show()

# Now to create a canny binary warped image using our M transform

canny_warped = cv2.warpPerspective(img_test_lane_canny1, M, img_size)

histogram = np.sum(canny_warped, axis=0)
width = len(histogram)
midpoint = width // 2

left_index_histogram = np.argmax(histogram[:midpoint])
right_index_histogram = np.argmax(histogram[midpoint:])

print(histogram, "histogram")
spike_coordinates = [left_index_histogram, right_index_histogram  + midpoint]
print(spike_coordinates[0], spike_coordinates[1],"Left and Right index")

leftx_current = spike_coordinates[0]
rightx_current = spike_coordinates[1]
margin = 100

plt.imshow(canny_warped)
plt.show()

nonzero = canny_warped.nonzero()
nonzero_y = np.array(nonzero[0])
nonzero_x = np.array(nonzero[1])

minpix = 50

# ===============================================================
# 9 Windows Lane Detection (Find Pixels & Draw Visualization)
# ===============================================================

# 1. Setup the windows and lists
nwindows = 9
window_height = int(h // nwindows)
nonzero = canny_warped.nonzero()
nonzero_y = np.array(nonzero[0])
nonzero_x = np.array(nonzero[1])

# Initialize empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Create an output image to draw on and visualize the result
out_img = np.dstack((canny_warped, canny_warped, canny_warped)) * 255

# 2. Loop through the windows
for i in range(nwindows):
    # Identify window boundaries
    win_y_low = h - (i * window_height)       # Bottom edge (larger y)
    win_y_high = h - ((i + 1) * window_height) # Top edge (smaller y)
    
    # Left Lane Boundaries
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    
    # Right Lane Boundaries
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # DRAW THE WINDOWS (Green for Left, Blue for Right) on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_high), (win_xleft_high, win_y_low), (0, 255, 0), 2) 
    cv2.rectangle(out_img, (win_xright_low, win_y_high), (win_xright_high, win_y_low), (0, 0, 255), 2) 

    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzero_y >= win_y_high) & (nonzero_y < win_y_low) & 
                      (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzero_y >= win_y_high) & (nonzero_y < win_y_low) & 
                       (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
    
    # Append indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    
    # Recenter next window if we found enough pixels
    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzero_x[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzero_x[good_right_inds]))

# 3. Concatenate the arrays of indices (Flattening the list)
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# 4. Extract pixel coordinates
leftx = nonzero_x[left_lane_inds]
lefty = nonzero_y[left_lane_inds] 
rightx = nonzero_x[right_lane_inds]
righty = nonzero_y[right_lane_inds]

# Visualize the final result with the boxes
plt.figure(figsize=(10, 7))
plt.imshow(out_img)
plt.title("Sliding Window Search")
plt.show()