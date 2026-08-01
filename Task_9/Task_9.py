import cv2
import matplotlib.pyplot as plt
import numpy as np


K_NOISE_SIZE = (5, 5)    
K_HOLE_SIZE = (11, 11)   
IMAGE_PATH = "1.jpg"     


def validate(mask_img, stage_name):
   
    total_pixels = mask_img.size
    white_pixels = cv2.countNonZero(mask_img)
    coverage = (white_pixels / total_pixels) * 100

    contours, _ = cv2.findContours(
        mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    num_regions = len(contours)

    print(f"[{stage_name:20s}] White Px: {white_pixels:6d} | Coverage: {coverage:5.1f}% | Regions: {num_regions}")
    return num_regions


def run_part_a():
    img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    rect_ops = [
        cv2.erode(mask, rect_kernel),
        cv2.dilate(mask, rect_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect_kernel)
    ]
    
    ellipse_ops = [
        cv2.erode(mask, ellipse_kernel),
        cv2.dilate(mask, ellipse_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    op_names = ["Erosion", "Dilation", "Opening", "Closing"]

    for col in range(4):
        axes[0, col].imshow(rect_ops[col], cmap="gray")
        axes[0, col].set_title(f"3x3 RECT | {op_names[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(ellipse_ops[col], cmap="gray")
        axes[1, col].set_title(f"5x5 ELLIPSE | {op_names[col]}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("part_a_morph_grid.png", dpi=150)
    plt.show()

    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, rect_kernel)
    overlay = img.copy()
    overlay[gradient > 0] = [0, 255, 0] 

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask, cmap="gray"); axes[0].set_title("Binary Mask"); axes[0].axis("off")
    axes[1].imshow(gradient, cmap="gray"); axes[1].set_title("Gradient Border"); axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[2].set_title("Green Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("part_a_gradient_overlay.png", dpi=150)
    plt.show()

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Grayscale Input"); axes[0].axis("off")
    axes[1].imshow(tophat, cmap="gray"); axes[1].set_title("Top-Hat (Bright Details)"); axes[1].axis("off")
    axes[2].imshow(blackhat, cmap="gray"); axes[2].set_title("Black-Hat (Dark Details)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("part_a_hat_transforms.png", dpi=150)
    plt.show()


def run_part_b():
    img = cv2.imread(IMAGE_PATH)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_r1, upper_r1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    lower_r2, upper_r2 = np.array([160, 100, 50]), np.array([179, 255, 255])
    raw_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), cv2.inRange(hsv, lower_r2, upper_r2))

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_NOISE_SIZE)
    mask_open = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open, iterations=2)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_HOLE_SIZE)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, k_close, iterations=2)

    
    print("\n--- Part B Pipeline Validation ---")
    reg_raw = validate(raw_mask, "Raw Mask")
    reg_open = validate(mask_open, "After Opening")
    reg_clean = validate(mask_clean, "After Closing")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stages = [raw_mask, mask_open, mask_clean]
    titles = ["Raw Mask", "After Opening (5x5)", "After Closing (11x11)"]

    for ax, m, t in zip(axes, stages, titles):
        ax.imshow(m, cmap="gray")
        ax.set_title(t)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mask_cleanup.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_part_a()
    run_part_b()