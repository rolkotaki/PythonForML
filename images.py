import cv2
import numpy as np
from matplotlib import pyplot


# Loading Images

# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# showing the image
# pyplot.imshow(image, cmap="gray"), pyplot.axis("off")
# pyplot.show()

print(type(image))  # We have transformed the image into a matrix whose elements correspond to individual pixels
print(image)
# In grayscale images, the value of an individual element is the pixel intensity, ranging from black (0) to white (255)
print(image.shape)  # resolution of our image was 3600 × 2270

# Load image in color
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Show pixel
print(image_bgr[0, 0])  # in case of coloured pictures, each element contains three values corresponding to BGR (RGB)
#  OpenCV uses BGR, not RGB. To properly display OpenCV color images in Matplotlib, we need to first convert it:
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Show image
# pyplot.imshow(image_rgb), pyplot.axis("off")
# pyplot.show()


# Saving Images

cv2.imwrite("images/plane_new.jpg", image_bgr)  # it will overwrite existing files without asking for confirmation


# Resizing Images

image_50x50 = cv2.resize(image, (50, 50))  # image: loaded before
# View image
# pyplot.imshow(image_50x50, cmap="gray"), pyplot.axis("off")
# pyplot.show()


# Cropping Images

image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Select first half of the columns and all rows
image_cropped = image[:, : 128]
# pyplot.imshow(image_cropped, cmap="gray"), pyplot.axis("off")
# pyplot.show()


# Blurring Images

image_blurry = cv2.blur(image, (5, 5))  # 40, 40 - very blurry
# Show image
# pyplot.imshow(image_blurry, cmap="gray"), pyplot.axis("off")
# pyplot.show()

# To blur an image, each pixel is transformed to be the average value of its neighbors.
# This neighbor and the operation performed are mathematically represented as a kernel.
# The size of this kernel determines the amount of blurring, with larger kernels producing smoother images.
# Here we blur an image by averaging the values of a 5 × 5 kernel around each pixel.

# Kernels are widely used in image processing to do everything from sharpening to edge detection
kernel = np.ones((5, 5)) / 25.0
print(kernel)
# We can manually apply a kernel to an image using filter2D to produce a similar blurring effect
# Since all elements have the same value (normalized to add up to 1), each has an equal say in the result
image_kernel = cv2.filter2D(image, -1, kernel)
# pyplot.imshow(image_kernel, cmap="gray"), pyplot.xticks([]), pyplot.yticks([])
# pyplot.show()


# Sharpening Images

image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Create kernel
kernel = np.array([[0, -1,  0],
                   [-1, 5, -1],
                   [0, -1,  0]])  # a kernel to highlight the pixel itself
# Sharpen image
image_sharp = cv2.filter2D(image, -1, kernel)
# pyplot.imshow(image_sharp, cmap="gray"), pyplot.axis("off")
# pyplot.show()


# Enhancing Contrast

image_enhanced = cv2.equalizeHist(image)  # Histogram equalization
# When we have a grayscale image, we can apply OpenCV’s equalizeHist directly on the image
# pyplot.imshow(image_enhanced, cmap="gray"), pyplot.axis("off")
# pyplot.show()

# When we have a color image, we first need to convert the image to the YUV color format.
# The Y is the luma, or brightness, and U and V denote the color. After the conversion, we can apply equalizeHist
# to the image and then convert it back to BGR or RGB.

# Load image
image_bgr = cv2.imread("images/plane.jpg")
# Convert to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# Apply histogram equalization
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
# Convert to RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
# Show image
# pyplot.imshow(image_rgb), pyplot.axis("off")
# pyplot.show()

# If histogram equalization is able to make objects of interest more distinguishable from other objects or backgrounds
# (which is not always the case), then it can be a valuable addition to our image pre-processing pipeline.


# Isolating Colors

image_bgr = cv2.imread('images/plane_256x256.jpg')
# Convert BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# Define range of blue values in HSV
lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])
# Create mask
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# Mask image
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# Show image
# pyplot.imshow(image_rgb), pyplot.axis("off")
# pyplot.show()
# Isolating colors in OpenCV is straightforward. First we convert an image into HSV (hue, saturation, and value).
# Second, we define a range of values we want to isolate, which is probably the most difficult and time-consuming part.
# Third, we create a mask for the image (we will only keep the white areas):
# pyplot.imshow(mask, cmap='gray'), pyplot.axis("off")
# pyplot.show()


# Binarizing Images

# Thresholding: pixels with intensity greater than some value to be white and less than the value to be black.
# Adaptive thresholding: where the threshold value for a pixel is determined by the pixel intensities of its neighbors.
# This can be helpful when lighting conditions change over different regions in an image.

image_grey = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Apply adaptive thresholding
max_output_value = 255   # the maximum intensity of the output pixel intensities
neighborhood_size = 99   # the size of the neighborhood used to determine a pixel’s threshold
subtract_from_mean = 10  # constant subtracted from the calculated threshold (used to manually fine-tune the threshold)
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: sets a pixel’s threshold to be a weighted sum of the neighboring pixel intensities
# cv2.ADAPTIVE_THRESH_MEAN_C: would set the threshold to simply the mean of the neighboring pixels
# pyplot.imshow(image_binarized, cmap="gray"), pyplot.axis("off")
# pyplot.show()


# Removing Backgrounds

image_bgr = cv2.imread('images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)
# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# Run grabCut
cv2.grabCut(image_rgb,  # Our image
            mask,       # The Mask
            rectangle,  # Our rectangle
            bgdModel,   # Temporary array for background
            fgdModel,   # Temporary array for background
            5,          # Number of iterations
            cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle
# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# pyplot.imshow(image_rgb_nobg), pyplot.axis("off")
# pyplot.show()

# Show mask
# pyplot.imshow(mask, cmap='gray'), pyplot.axis("off")
# pyplot.show()
# The black region is the area outside our rectangle that is assumed to be definitely background.
# The gray area is what GrabCut considered likely background, while the white area is likely foreground.
# The second mask is then applied to the image so that only the foreground remains.
# pyplot.imshow(mask_2, cmap='gray'), pyplot.axis("off")
# pyplot.show()


# Detecting Edges

image_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Calculate median intensity
median_intensity = np.median(image_gray)
# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# Apply canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
# Show image
# pyplot.imshow(image_canny, cmap="gray"), pyplot.axis("off")
# pyplot.show()


# Detecting Corners

image_bgr = cv2.imread("images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)
# Set corner detector parameters
block_size = 2  # size of the neighbor around each pixel used for corner detection
aperture = 29   # size of the Sobel kernel used
free_parameter = 0.04  # free parameter where larger values correspond to identifying softer corners
# Detect corners
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)
# Large corner markers
detector_responses = cv2.dilate(detector_responses, None)
# Only keep detector responses greater than threshold, mark as white
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255, 255, 255]
# Convert to grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# Show image
# pyplot.imshow(image_gray, cmap="gray"), pyplot.axis("off")
# pyplot.show()
# Show potential corners
# pyplot.imshow(detector_responses, cmap='gray'), pyplot.axis("off")
# pyplot.show()

# alternative way to detect corners: cv2.goodFeaturesToTrack


# Creating Features for Machine Learning

image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))
# Convert image data to one-dimensional vector
image_10x10.flatten()
# If an image is in grayscale, each pixel is presented by one value (i.e., pixel intensity: 1 if white, 0 if black).
print(image_10x10.shape)
print(image_10x10.flatten().shape)

# If the image is in color, it is represented by multiple values (most often three) representing the channels (ie: RGB).
# If our 10 × 10 image is in color, we will have 300 feature values for each observation.
image_color = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)
# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))
# Convert image data to one-dimensional vector, show dimensions
print(image_color_10x10.flatten().shape)


# Encoding Mean Color as a Feature

image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)
# Calculate the mean of each channel
channels = cv2.mean(image_bgr)
# Swap blue and red values (making it RGB, not BGR)
observation = np.array([(channels[2], channels[1], channels[0])])
# Show mean channel values
print(observation)
# pyplot.imshow(observation), pyplot.axis("off")
# pyplot.show()


# Encoding Color Histograms as Features

image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)
# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Create a list for feature values
features = []
# Calculate the histogram for each color channel
colors = ("r", "g", "b")
# For each channel: calculate histogram and add to feature value list
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # Image
                             [i],       # Index of channel
                             None,      # No mask
                             [256],     # Histogram size
                             [0, 256])  # Range
    features.extend(histogram)
# Create a vector for an observation's feature values
observation = np.array(features).flatten()
# Show the observation's value for the first five features
print(observation[0:5])

# For each channel: calculate histogram, make plot
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # Image
                             [i],       # Index of channel
                             None,      # No mask
                             [256],     # Histogram size
                             [0, 256])  # Range
    pyplot.plot(histogram, color=channel)
    pyplot.xlim([0, 256])
# Show plot
# pyplot.show()
