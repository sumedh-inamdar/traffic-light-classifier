import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):

    ## TODO: Resize image and pre-process so that all "standard" images are the same size
    standard_im = np.copy(image)
    resize_im = cv2.resize(standard_im, (32, 32))

    return resize_im

## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples:
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]
def one_hot_encode(label):

    color_map = ["red", "yellow", "green"]
    one_hot = [0] * len(color_map)
    try:
        one_hot[color_map.index(label)] = 1
        return one_hot
    except:
        raise TypeError('Please input red, yellow, or green. Not ', label)


def standardize(image_list):

    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):

    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_mask = apply_mask(hsv)
    hsv_cropd = apply_crop(hsv_mask)

    h = hsv_cropd[:,:,0]
    s = hsv_cropd[:,:,1]
    v = hsv_cropd[:,:,2]
    ## TODO: Create and return a feature array (sum of brightness by rows)
    feature = []
    numRows = len(v)
    numCols = len(v[0])
    for row in range(numRows):
        rowTotal = 0
        for col in range(numCols):
            #sum value by row and append to feature list
            rowTotal = rowTotal + v[row][col]
        feature.append(rowTotal)
    featureByRegion = combinedValues(feature)
    return (featureByRegion, hsv_cropd)


def apply_mask(hsv_image):
    lower_h = np.array([0,0,120])
    upper_h = np.array([180,255,255])
    ## TODO: Define the masked area and mask the image
    # Don't forget to make a copy of the original image to manipulate
    hsv_mask = cv2.inRange(hsv_image, lower_h, upper_h)
    masked_image = np.copy(hsv_image)
    masked_image[hsv_mask == 0] = [0, 0, 0]
    return masked_image

def apply_crop(hsv_masked):
    # Make a copy of the image to manipulate
    image_crop = np.copy(hsv_masked)
    # Define how many pixels to slice off the sides of the original image
    top_row_crop = 5
    bottom_row_crop = 2
    col_crop = 7
    # Using image slicing, subtract the row_crop from top/bottom and col_crop from left/right
    image_crop = hsv_masked[top_row_crop:-bottom_row_crop, col_crop:-col_crop, :]
    return image_crop
#This function takes in the feature array (broken down by rows) and combines the values evenly between three rows.
#Returns an array of three values.

def combinedValues(featureByRow):
    edge = len(featureByRow)/3
    edgeDown = int(np.floor(edge))
    edgeUp = edgeDown + 1
    remainder = edge%1
    edgeDown2 = 2*edgeDown + int(np.floor(2*remainder))
    edgeUp2 = edgeDown2 + 1
    rem2 = (2*edge)%1

    #sum the saturation values by thirds and return the index of the max value
    binnedSValue = [sum(featureByRow[0:edgeDown]) + remainder*featureByRow[edgeDown],
                    (1-remainder)*featureByRow[edgeDown] + sum(featureByRow[edgeUp:edgeDown2]) + rem2*featureByRow[edgeDown2]
                    , (1-rem2)*featureByRow[edgeDown2] + sum(featureByRow[edgeUp2:])]
    return binnedSValue

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):

    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    featureByRegion = create_feature(rgb_image)[0]
    numberlabel = np.argmax(featureByRegion) #f
    if numberlabel == 0:
        return one_hot_encode('red')
    elif numberlabel == 1:
        return one_hot_encode('yellow')
    return one_hot_encode('green')

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

image = 1

(missed_image_original, predicted_label, true_label) = MISCLASSIFIED[image]

hsv = cv2.cvtColor(missed_image_original, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the misclassifed image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Original Misclassified image')
ax1.imshow(missed_image_original)
ax2.set_title('H channel')
ax2.imshow(h, cmap ='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')
plt.show()
