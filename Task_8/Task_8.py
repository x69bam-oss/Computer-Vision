import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

T_value = 127

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

__,thresh_binary = cv2.threshold(gray, T_value, 255, cv2.THRESH_BINARY)
__,thresh_binary_inv = cv2.threshold(gray, T_value, 255, cv2.THRESH_BINARY_INV)
__,thresh_trunc = cv2.threshold(gray, T_value, 255, cv2.THRESH_TRUNC)
__,thresh_tozero = cv2.threshold(gray, T_value, 255, cv2.THRESH_TOZERO)
__,thresh_tozero_inv = cv2.threshold(gray, T_value, 255, cv2.THRESH_TOZERO_INV)

fig , axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(thresh_binary, cmap='gray')
axes[0, 1].set_title('Binary Threshold')
axes[0, 2].imshow(thresh_binary_inv, cmap='gray')
axes[0, 2].set_title('Binary Inverse Threshold')
axes[1, 0].imshow(thresh_trunc, cmap='gray')
axes[1, 0].set_title('Truncated Threshold')
axes[1, 1].imshow(thresh_tozero, cmap='gray')
axes[1, 1].set_title('To Zero Threshold')
axes[1, 2].imshow(thresh_tozero_inv, cmap='gray')
axes[1, 2].set_title('To Zero Inverse Threshold')
axes[0, 0].axis('off')
axes[0, 1].axis('off')
axes[0, 2].axis('off')
axes[1, 0].axis('off')
axes[1, 1].axis('off')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('thresholding_results.png')  
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
otsu_val, otsu_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu's threshold value: {otsu_val}")

fig , axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(blurred, cmap='gray')
axes[0].set_title('Blurred Image')
axes[1].imshow(otsu_img, cmap='gray')
axes[1].set_title("Otsu's Thresholding")
axes[0].axis('off')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('otsu_thresholding_results.png')
plt.show()

block_size = 11
adaptive_mean_c2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)
adaptive_gaussian_c2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

adaptive_mean_c6 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 5)
adaptive_gaussian_c6 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5)

fig , axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(adaptive_mean_c2, cmap='gray')
axes[0, 0].set_title('Adaptive Mean Thresholding (C=2)')
axes[0, 1].imshow(adaptive_gaussian_c2, cmap='gray')
axes[0, 1].set_title('Adaptive Gaussian Thresholding (C=2)')
axes[1, 0].imshow(adaptive_mean_c6, cmap='gray')
axes[1, 0].set_title('Adaptive Mean Thresholding (C=6)')
axes[1, 1].imshow(adaptive_gaussian_c6, cmap='gray')
axes[1, 1].set_title('Adaptive Gaussian Thresholding (C=6)')
axes[0, 0].axis('off')
axes[0, 1].axis('off')
axes[1, 0].axis('off')
axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig('adaptive_thresholding_results.png')
plt.show()

def Validate_Mask(mask_img) :
    total_pixels = mask_img.size
    white_pixels = np.sum(mask_img == 255)
    coverage_percentage = (white_pixels / total_pixels) * 100
    
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    print("\n==========================================")
    print("      --- MASK VALIDATION RESULTS ---     ")
    print("==========================================")
    print(f"White Pixel Count   : {white_pixels}")
    print(f"Coverage Percentage : {coverage_percentage:.2f}%")
    print(f"Separate Regions    : {num_contours}")
    print("==========================================\n")
    
    return white_pixels, coverage_percentage, num_contours

lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise Exception("Could not open video device")
else:
    print("\n--- Live Camera Segmentation Running ---")
    print("-> Press 's' to Capture best state & Run validate_mask()")
    print("-> Press 'q' to Quit\n")
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    result_frame = cv2.bitwise_and(frame, frame, mask=cleaned_mask)
    
    cv2.imshow("Raw Frame (Live)", frame)
    cv2.imshow("Binary Mask (Live)", cleaned_mask)
    cv2.imshow("Masked Result (Live)", result_frame)
    
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('s'):
        cv2.imwrite('captured_frame.jpg', frame)
        cv2.imwrite('captured_mask.jpg', cleaned_mask)
        cv2.imwrite('captured_result.jpg', result_frame)
        print("saved captured_frame.jpg, captured_mask.jpg, and captured_result.jpg")
        
        validate_Mask(cleaned_mask)
        break
    elif key == ord('q'):
        break
    
    cap.release()
    cv2.destroyAllWindows()
        