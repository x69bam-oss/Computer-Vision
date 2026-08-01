import cv2
import numpy as np


rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))


diamond_kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=np.uint8)


print("--- Rectangular Kernel (cv2.MORPH_RECT) 5x5 ---")
print(rect_kernel)

print("\n--- Elliptical Kernel (cv2.MORPH_ELLIPSE) 5x5 ---")
print(ellipse_kernel)

print("\n--- Cross-Shaped Kernel (cv2.MORPH_CROSS) 5x5 ---")
print(cross_kernel)

print("\n--- Custom Diamond-Shaped Kernel (NumPy uint8) 5x5 ---")
print(diamond_kernel)