import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("1.jpg") 
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  


magnitude = cv2.magnitude(sobelX, sobelY)


abs_sobelX = cv2.convertScaleAbs(sobelX)
abs_sobelY = cv2.convertScaleAbs(sobelY)
sobel_mag_u8 = cv2.convertScaleAbs(magnitude)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(abs_sobelX, cmap="gray")
axes[0, 1].set_title("Sobel X (Vertical Edges)")
axes[0, 1].axis("off")

axes[1, 0].imshow(abs_sobelY, cmap="gray")
axes[1, 0].set_title("Sobel Y (Horizontal Edges)")
axes[1, 0].axis("off")

axes[1, 1].imshow(sobel_mag_u8, cmap="gray")
axes[1, 1].set_title("Sobel Magnitude Combined")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("part_a_step2_sobel_grid.png", dpi=150)
plt.show()


lap_noisy = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_noisy_u8 = cv2.convertScaleAbs(lap_noisy)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
lap_clean = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
lap_clean_u8 = cv2.convertScaleAbs(lap_clean)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(lap_noisy_u8, cmap="gray")
axes[0].set_title("Raw Laplacian (No Blur - Highly Noisy)")
axes[0].axis("off")

axes[1].imshow(lap_clean_u8, cmap="gray")
axes[1].set_title("Laplacian of Gaussian - LoG (With 5x5 Blur)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("part_a_step3_laplacian_compare.png", dpi=150)
plt.show()


canny_pair1 = cv2.Canny(blurred, 30, 90)    
canny_pair2 = cv2.Canny(blurred, 50, 150)    
canny_pair3 = cv2.Canny(blurred, 100, 300)  


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

canny_imgs = [canny_pair1, canny_pair2, canny_pair3]
canny_titles = [
    "Canny Low Pair (30, 90) - Cluttered",
    "Canny Standard Pair (50, 150) - Optimal",
    "Canny High Pair (100, 300) - Structural Only"
]

for ax, im, title in zip(axes, canny_imgs, canny_titles):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("part_a_step4_canny_ratios.png", dpi=150)
plt.show()


canny_raw = cv2.Canny(gray, 50, 150)


canny_blurred = cv2.Canny(blurred, 50, 150)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(canny_raw, cmap="gray")
axes[0].set_title("Canny without Blur (Raw - False Edges)")
axes[0].axis("off")

axes[1].imshow(canny_blurred, cmap="gray")
axes[1].set_title("Canny with Gaussian Blur (Clean)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("part_a_step5_canny_blur_effect.png", dpi=150)
plt.show()



frame = cv2.imread("1.jpg")
if frame is None:
    raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")


BLUR_K = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_COLOR = (0, 255, 0)  

blurred_frame = cv2.GaussianBlur(frame, BLUR_K, 0)
gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray_frame, CANNY_LOW, CANNY_HIGH)


output_overlay = frame.copy()
output_overlay[edges == 255] = EDGE_COLOR


cv2.imwrite("edge_raw.jpg", frame)
cv2.imwrite("edge_detected.png", edges)
cv2.imwrite("edge_overlay.jpg", output_overlay)
print("\n Successfully saved Part B outputs: edge_raw.jpg, edge_detected.png, edge_overlay.jpg")


edge_pixel_count = np.count_nonzero(edges)
total_pixels = edges.size
edge_coverage_pct = (edge_pixel_count / total_pixels) * 100

print("\n==========================================")
print("      --- TASK 10 PART B RESULTS ---      ")
print("==========================================")
print(f"Edge Pixel Count   : {edge_pixel_count}")
print(f"Edge Coverage      : {edge_coverage_pct:.2f}%")
print("==========================================\n")


edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
combined_display = np.hstack([frame, edge_bgr, output_overlay])

cv2.imshow("Part B: Raw | Edges | Green Overlay", combined_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

