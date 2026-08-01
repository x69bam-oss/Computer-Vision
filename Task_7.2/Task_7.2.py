import cv2
import matplotlib.pyplot as plt
import numpy as np


img_bgr = cv2.imread("1.jpg")

if img_bgr is None:
    raise FileNotFoundError("Image file '1.jpg' not found. Please check the path and filename.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

T = 127  # Threshold value for manual thresholding

_, thresh_binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc = cv2.threshold(gray, T, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(gray, T, 255, cv2.THRESH_TOZERO)
_, thresh_tozero_inv = cv2.threshold(gray, T, 255, cv2.THRESH_TOZERO_INV)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)
otsu_val, thresh_otsu = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


thresh_adaptive_gauss = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)


titles = [
    "1. Original Image",
    f"2. Binary (T={T})",
    f"3. Binary Inverse (T={T})",
    f"4. Truncate (T={T})",
    f"5. To Zero (T={T})",
    f"6. To Zero Inv (T={T})",
    f"7. Otsu's (Auto T={int(otsu_val)})",
    "8. Adaptive Gaussian",
]

images = [
    img_rgb,
    thresh_binary,
    thresh_binary_inv,
    thresh_trunc,
    thresh_tozero,
    thresh_tozero_inv,
    thresh_otsu,
    thresh_adaptive_gauss,
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, (ax, im, title) in enumerate(zip(axes.flatten(), images, titles)):
    if idx == 0:
        ax.imshow(im)
    else:
        ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("thresholding_comparison_single_window.png", dpi=150)
plt.show()


print(f"Otsu Automatically Found T-Value: {otsu_val:.1f}")

