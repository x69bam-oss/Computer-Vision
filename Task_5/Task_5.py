import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.png')

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

mean_value = gray.mean()
std_value = gray.std()

print("--- Part A - Step 1: Diagnosis Stats ---")
print(f"Mean Brightness: {mean_value:.2f} (0=Black, 255=White)")
print(f"Standard Deviation: {std_value:.2f} (Low = Low Contrast)")

if mean_value < 80:
    print("Diagnosis: The image is too dark.")
elif mean_value > 180:
    print("Diagnosis: The image is too bright.")
elif std_value < 30:    
    print("Diagnosis: The image has low contrast.")
else:
    print("Diagnosis: The image has acceptable brightness and contrast.")
    
plt.figure(figsize=(10, 5))
plt.plot(hist_gray.flatten(), color='black', linewidth=1.5)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.tight_layout()
plt.savefig('histogram.png')
plt.show()


colors = ('b', 'g', 'r')
chhannels_name = ('Blue', 'Green', 'Red')

plt.figure(figsize=(10, 5))
for i, color in enumerate(colors):
    hist_color = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_color.flatten(), color=color, linewidth=1.5, label=f'{chhannels_name[i]} Channel')
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.tight_layout()
plt.savefig('color_histogram.png')
plt.show()

clip_limit = 2.0
title_grid = (8 , 8)

eq_gray =cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid)
clahe_gray = clahe.apply(gray)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

L__enchanced = clahe.apply(l)
lab_enhanced = cv2.merge((L__enchanced, a, b))

clahe_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

hist_orginal = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])
hist_clahe = cv2.calcHist([clahe_gray], [0], None, [256], [0, 256])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

imags = [gray, eq_gray, clahe_gray]
hists = [hist_orginal, hist_eq, hist_clahe]
titles = ['Original Grayscale Image', 'Histogram Equalized Image', 'CLAHE Enhanced Image']
colors = ['black', 'teal', 'darkgreen']

for col  in range(3):
    axes[0,col].imshow(imags[col], cmap='gray')
    axes[0,col].set_title(titles[col])
    axes[0,col].axis('off')
    
    axes[1,col].plot(hists[col].flatten(), color=colors[col], linewidth=1.5)
    axes[1,col].set_title(f'{titles[col]} Histogram')
    axes[1,col].set_xlabel('Pixel Intensity')
    axes[1,col].set_ylabel('Frequency')
    axes[1,col].set_xlim([0, 256])
    axes[1,col].fill_between(range(256), hists[col].flatten(), color=colors[col], alpha=0.5)

plt.tight_layout()
plt.suptitle('Image Enhancement and Histogram Comparison', fontsize=16)
plt.savefig('enhancement_comparison.png')
plt.show()
