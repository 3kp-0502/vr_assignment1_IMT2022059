# Coin Detection and Image Stitching

**Author:** Pudi Kireeti (IMT2022059)

## Overview

This project consists of two main parts:

1. **Coin Detection, Segmentation, and Counting**
   - Detect coins using edge detection techniques
   - Segment individual coins using region-based segmentation
   - Count the total number of detected coins

2. **Image Stitching for Panorama Creation**
   - Detect key points in overlapping images
   - Use homography to align and stitch images into a panorama

## Requirements

Install the required dependencies using:

```
pip install opencv-python numpy matplotlib
```

## Repository Structure

```
vr_assignment1_IMT2022059/
├── input_images/
│   ├── coins.jpg
│   ├── left.jpg
│   └── right.jpg
├── output_images/
│   ├── blurred.jpg
│   ├── closed.jpg
│   ├── detected_coins.jpg
│   ├── edges.jpg
│   ├── gray.jpg
│   ├── keypoints_left.jpg
│   ├── keypoints_right.jpg
│   ├── matches.jpg
│   ├── panorama.jpg
│   └── segmented_coin_*.jpg (multiple files)
├── 1.py (Coin Detection & Segmentation)
├── 2.py (Panorama Stitching)
└── README.md
```

## Part 1: Coin Detection, Segmentation, and Counting

This section detects, segments, and counts Indian coins in an image.

### Steps Involved

1. **Edge Detection (Contours Method):**
   - Convert the image to grayscale using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
   - Apply Gaussian Blur and Canny edge detection using `cv2.GaussianBlur` and `cv2.Canny`
   - Find contours to detect coin boundaries using `cv2.findContours`
   - Outline detected coins in green using `cv2.drawContours`

2. **Segmentation:**
   - Filter valid coins based on area and circularity criteria using the `filter_valid_coins` function
   - Create masks for each valid coin using `cv2.drawContours` with `thickness=cv2.FILLED`
   - Extract individual coin segments using the masks with `cv2.bitwise_and`

3. **Counting Coins:**
   - Determine the total number of detected coins by counting valid contours using `len(valid_coins)`
   - Display the final count in the terminal

### Methods Used

- **Preprocessing:** Grayscale conversion and Gaussian blur to reduce noise
- **Edge Detection:** Canny edge detection with optimized thresholds for coin boundaries
- **Contour Detection:** Identifying closed shapes that represent coins
- **Filtering:** Using area and circularity metrics to eliminate non-coin objects
- **Segmentation:** Creating masks for each coin and extracting individual segments

### How to Run

Place your input image in the `input_images/` folder and update the image path in `1.py`.

Then run:
```
python 1.py
```

### Implementation Details

The successful implementation uses these optimized parameters:

1. **Preprocessing**
   - Smaller Gaussian blur kernel (7,7) to preserve fine edges

2. **Edge Detection Parameters**
   - Lower Canny thresholds (20, 150) to detect faint edges of coins

3. **Morphological Operations**
   - Smaller kernel (3x3) for `cv2.morphologyEx` to preserve fine details

4. **Coin Filtering Criteria**
   - Area: 800 to 60,000
   - Circularity: > 0.6
   - These thresholds allow detection of slightly irregular or small coins

## Part 2: Image Stitching

This section stitches overlapping images into a single panorama.

### Steps Involved

1. **Feature Detection & Matching:**
   - Use SIFT (Scale-Invariant Feature Transform) to extract keypoints with `cv2.SIFT_create()` and `sift.detectAndCompute()`
   - Match keypoints using FLANN (Fast Library for Approximate Nearest Neighbors) with `cv2.FlannBasedMatcher()` and `flann.knnMatch()`
   - Filter matches based on a ratio test to keep only good matches (m.distance < 0.7 * n.distance)

2. **Homography & Warping:**
   - Compute the homography matrix using RANSAC algorithm with `cv2.findHomography(..., cv2.RANSAC, ...)`
   - Warp the left image to align with the right image using `cv2.warpPerspective()`
   - Create a seamless panorama by combining the warped images

### Methods Used

- **Feature Detection:** SIFT algorithm to find distinctive keypoints
- **Matching:** FLANN-based matcher with KD-tree for efficient matching (FLANN_INDEX_KDTREE)
- **Filtering:** Ratio test to eliminate poor matches
- **Homography:** RANSAC method to find the optimal transformation matrix
- **Warping:** Perspective transformation to align and combine images

### How to Run

Place overlapping images in the `input_images/` folder and update the image paths in `2.py`.

Then run:
```
python 2.py
```

### Implementation Details

The successful implementation uses these optimized parameters:

1. **Keypoint Detection**
   - SIFT detector for robust scale and rotation invariant features

2. **Keypoint Matching**
   - FLANN-based matching with correct settings (trees=5, checks=50)
   - Lowe's Ratio Test (m.distance < 0.7 * n.distance) to filter good matches

3. **Homography Estimation**
   - Dynamic homography matrix using `cv2.findHomography` with RANSAC to eliminate outliers

4. **Image Warping & Blending**
   - Proper perspective warping based on the computed homography matrix
   - Correct placement of the right image for seamless stitching

## Results & Observations

### Coin Detection & Segmentation
- Successfully detected coin edges using Canny edge detection with optimized parameters
- Effectively filtered valid coins using area and circularity criteria
- Segmented individual coins using contour-based masks
- The algorithm correctly identified and counted the coins in the test image

### Image Stitching
- SIFT keypoints were accurately detected in both input images
- FLANN-based matching successfully identified corresponding points between images
- Homography with RANSAC effectively handled perspective differences
- The final panorama seamlessly combined both images with smooth transitions

## GitHub Repository
[3kp-0502/vr_assignment1_IMT2022059](https://github.com/3kp-0502/vr_assignment1_IMT2022059)
