import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
left_img = cv2.imread("C:/Users/3kp05/Downloads/left.jpg")
right_img = cv2.imread("C:/Users/3kp05/Downloads/right.jpg")

# Convert to grayscale
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Step 1: Detect keypoints using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Draw keypoints with a different color (Blue)
keypoint_img1 = cv2.drawKeypoints(left_img, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(right_img, kp2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Save keypoint images
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_left.jpg", keypoint_img1)
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_right.jpg", keypoint_img2)

# Display keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Left Image (Green)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Right Image (Green)")
plt.axis("off")
plt.show()

# Step 2: Match keypoints using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test to keep good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Draw matches
matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save matched keypoints image
cv2.imwrite("C:/Users/3kp05/Downloads/matches.jpg", matched_img)

# Display matched keypoints
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Keypoints")
plt.axis("off")
plt.show()

# Step 3: Find homography and warp
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions for the final stitched output
    height, width, channels = right_img.shape
    result = cv2.warpPerspective(left_img, H, (width * 2, height))
    result[0:height, 0:width] = right_img  # Place right image correctly

    # Save final panorama
    cv2.imwrite("C:/Users/3kp05/Downloads/panorama.jpg", result)

    # Display final panorama
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama")
    plt.axis("off")
    plt.show()

    print("Panorama saved")

else:
    print("Not enough matches found!")import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
left_img = cv2.imread("C:/Users/3kp05/Downloads/left.jpg")
right_img = cv2.imread("C:/Users/3kp05/Downloads/right.jpg")

# Convert to grayscale
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Step 1: Detect keypoints using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Draw keypoints with a different color (Blue)
keypoint_img1 = cv2.drawKeypoints(left_img, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(right_img, kp2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Save keypoint images
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_left.jpg", keypoint_img1)
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_right.jpg", keypoint_img2)

# Display keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Left Image (Green)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Right Image (Green)")
plt.axis("off")
plt.show()

# Step 2: Match keypoints using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test to keep good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Draw matches
matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save matched keypoints image
cv2.imwrite("C:/Users/3kp05/Downloads/matches.jpg", matched_img)

# Display matched keypoints
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Keypoints")
plt.axis("off")
plt.show()

# Step 3: Find homography and warp
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions for the final stitched output
    height, width, channels = right_img.shape
    result = cv2.warpPerspective(left_img, H, (width * 2, height))
    result[0:height, 0:width] = right_img  # Place right image correctly

    # Save final panorama
    cv2.imwrite("C:/Users/3kp05/Downloads/panorama.jpg", result)

    # Display final panorama
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama")
    plt.axis("off")
    plt.show()

    print("Panorama saved")

else:
    print("Not enough matches found!")import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
left_img = cv2.imread("C:/Users/3kp05/Downloads/left.jpg")
right_img = cv2.imread("C:/Users/3kp05/Downloads/right.jpg")

# Convert to grayscale
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Step 1: Detect keypoints using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Draw keypoints with a different color (Blue)
keypoint_img1 = cv2.drawKeypoints(left_img, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(right_img, kp2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Save keypoint images
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_left.jpg", keypoint_img1)
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_right.jpg", keypoint_img2)

# Display keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Left Image (Green)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Right Image (Green)")
plt.axis("off")
plt.show()

# Step 2: Match keypoints using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test to keep good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Draw matches
matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save matched keypoints image
cv2.imwrite("C:/Users/3kp05/Downloads/matches.jpg", matched_img)

# Display matched keypoints
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Keypoints")
plt.axis("off")
plt.show()

# Step 3: Find homography and warp
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions for the final stitched output
    height, width, channels = right_img.shape
    result = cv2.warpPerspective(left_img, H, (width * 2, height))
    result[0:height, 0:width] = right_img  # Place right image correctly

    # Save final panorama
    cv2.imwrite("C:/Users/3kp05/Downloads/panorama.jpg", result)

    # Display final panorama
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama")
    plt.axis("off")
    plt.show()

    print("Panorama saved")

else:
    print("Not enough matches found!")import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
left_img = cv2.imread("C:/Users/3kp05/Downloads/left.jpg")
right_img = cv2.imread("C:/Users/3kp05/Downloads/right.jpg")

# Convert to grayscale
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Step 1: Detect keypoints using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Draw keypoints with a different color (Blue)
keypoint_img1 = cv2.drawKeypoints(left_img, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(right_img, kp2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Save keypoint images
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_left.jpg", keypoint_img1)
cv2.imwrite("C:/Users/3kp05/Downloads/keypoints_right.jpg", keypoint_img2)

# Display keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Left Image (Green)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Right Image (Green)")
plt.axis("off")
plt.show()

# Step 2: Match keypoints using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test to keep good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Draw matches
matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save matched keypoints image
cv2.imwrite("C:/Users/3kp05/Downloads/matches.jpg", matched_img)

# Display matched keypoints
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Keypoints")
plt.axis("off")
plt.show()

# Step 3: Find homography and warp
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions for the final stitched output
    height, width, channels = right_img.shape
    result = cv2.warpPerspective(left_img, H, (width * 2, height))
    result[0:height, 0:width] = right_img  # Place right image correctly

    # Save final panorama
    cv2.imwrite("C:/Users/3kp05/Downloads/panorama.jpg", result)

    # Display final panorama
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama")
    plt.axis("off")
    plt.show()

    print("Panorama saved")

else:
    print("Not enough matches found!")