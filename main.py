import cv2
import numpy as np


# Adding Padding
def pad(image, width):
    np.pad(image, pad_width=width, mode='symmetric', reflect_type='even')
    return image


# Blur Gaussian Filter
def blur_filter(image, kernel_size=3):
    filter_image = np.zeros_like(image)
    padding = kernel_size // 2

    for rgb_canal in range(3):
        for i in range(padding, image.shape[0] - padding):
            for j in range(padding, image.shape[1] - padding):
                neighbors = image[i - padding:i + padding + 1, j - padding:j + padding + 1, rgb_canal]
                filter_image[i, j, rgb_canal] = np.mean(neighbors)

    return filter_image


# Median Gaussian Filter
def median_filter(image, kernel_size=3):
    filter_image = np.zeros_like(image)
    padding = kernel_size // 2

    for rgb_canal in range(3):
        for i in range(padding, image.shape[0] - padding):
            for j in range(padding, image.shape[1] - padding):
                neighbors = image[i - padding:i + padding + 1, j - padding:j + padding + 1, rgb_canal].flatten()
                filter_image[i, j, rgb_canal] = np.median(neighbors)

    return filter_image


# RGB to GrayScale
def rgb_to_gray(image):
    gray_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray_image[i, j] = np.sum(image[i, j]) // 3

    return gray_image


# FAST
def fast(image, threshold=20):

    def circle_check(pos_x, pos_y, circle):
        count_points = len(circle) // 4 * 3
        center = image[pos_x, pos_y]
        brighter = 0
        darker = 0
        for i, j in circle:
            current_pixel = image[pos_x + i, pos_y + j]
            if (current_pixel >= center + threshold).any():
                brighter += 1
            elif (current_pixel <= center - threshold).any():
                darker += 1
        if brighter >= count_points or darker >= count_points:
            return True

    points = []

    full_circle = [(0, -3), (1, -3), (2, -2), (3, -1),
                   (3, 0), (3, 1), (2, 2), (1, 3),
                   (0, 3), (-1, 3), (-2, 2), (-3, 1),
                   (-3, 0), (-3, -1), (-2, -2), (-1, -3)]

    fast_circle = [(0, -3), (3, 0), (0, 3), (-3, 0)]

    for x in range(3, image.shape[0] - 3):
        for y in range(3, image.shape[1] - 3):
            if circle_check(x, y, fast_circle):
                if circle_check(x, y, full_circle):
                    points.append((y, x))

    return points


# BRIEF descriptor
def brief(image, points):
    descriptors = np.array([])

    for i, j in points:
        descriptor = np.array([])
        for _ in range(32):
            i1, j1, i2, j2 = np.random.randint(-2, 3, size=4)
            if (0 <= i + i1 < image.shape[0]) and (0 <= j + j1 < image.shape[1]) and \
                    (0 <= i + i2 < image.shape[0]) and (0 <= j + j2 < image.shape[1]):
                pixel_comparison = image[i + i1, j + j1] < image[i + i2, j + j2]
                descriptor = np.append(descriptor, pixel_comparison)
        descriptors = np.append(descriptors, descriptor)

    return descriptors


# ORB
def orb(image):
    points = fast(image, threshold=20)
    descriptors = brief(image, points)

    return points, descriptors


if __name__ == '__main__':

    # INPUT
    original_image = cv2.imread('image.jpg')
    pad_image = pad(original_image, width=3)

    # FILTERS
    blur_image = blur_filter(pad_image, kernel_size=3)
    median_image = median_filter(blur_image, kernel_size=3)
    grayscale_image = rgb_to_gray(median_image)

    # FAST
    fast_image = np.array(original_image)
    fast_points = fast(grayscale_image, threshold=20)
    for point in fast_points:
        cv2.circle(fast_image, point, 6, (255, 255, 0), 1)

    # ORB
    orb_image = np.array(original_image)
    orb_points, orb_descriptor = orb(grayscale_image)
    for point in orb_points:
        cv2.circle(orb_image, point, 6, (0, 255, 255), 1)

    # OUTPUT
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Blur Image', blur_image)
    cv2.imshow('Median Image', median_image)
    cv2.imshow('Gray Image', grayscale_image)
    cv2.imshow('FAST Key Points', fast_image)
    cv2.imshow('ORB Key Points', orb_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
