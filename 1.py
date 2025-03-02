import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Loads and preprocesses the image by converting it to grayscale and applying Gaussian blur."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    cv2.imwrite("C:/Users/3kp05/Downloads/gray.jpg", gray)
    cv2.imwrite("C:/Users/3kp05/Downloads/blurred.jpg", blurred)
    
    return image, gray, blurred

def detect_coins(image, blurred):
    """Detects coin edges using Canny edge detection and morphological operations."""
    edges = cv2.Canny(blurred, 20, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imwrite("C:/Users/3kp05/Downloads/edges.jpg", edges)
    cv2.imwrite("C:/Users/3kp05/Downloads/closed.jpg", closed)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_valid_coins(contours):
    """Filters valid coins based on area and circularity criteria."""
    valid_coins = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

        if 800 < area < 60000 and circularity > 0.6:
            valid_coins.append(contour)
    return valid_coins

def segment_coins(image, valid_coins):
    """Segments individual coins using contour-based masks."""
    segmented_coins = []
    for i, contour in enumerate(valid_coins):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)
        segmented_coins.append(segmented_coin)
        cv2.imwrite(f"C:/Users/3kp05/Downloads/segmented_coin_{i+1}.jpg", segmented_coin)
    return segmented_coins

def display_results(image, valid_coins, segmented_coins):
    """Displays the detected coins and segmented coin images."""
    output_image = image.copy()
    cv2.drawContours(output_image, valid_coins, -1, (0, 255, 0), 3)
    cv2.imwrite("C:/Users/3kp05/Downloads/detected_coins.jpg", output_image)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Coins: {len(valid_coins)}")
    plt.axis("off")
    plt.show()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(segmented_coins):
            ax.imshow(cv2.cvtColor(segmented_coins[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Coin {i+1}")
        ax.axis("off")
    plt.show()

def main():
    """Main function to execute the coin detection pipeline."""
    image_path = "C:/Users/3kp05/Downloads/coins.jpg"
    image, gray, blurred = preprocess_image(image_path)
    contours = detect_coins(image, blurred)
    valid_coins = filter_valid_coins(contours)
    segmented_coins = segment_coins(image, valid_coins)
    display_results(image, valid_coins, segmented_coins)
    print(f"Final Coin Count: {len(valid_coins)}")

main()