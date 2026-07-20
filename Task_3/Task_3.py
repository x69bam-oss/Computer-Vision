import cv2
import numpy as np
import matplotlib.pyplot as plt

#reading image

img = cv2.imread('STR.jpg')

#checking Image 

if img is None :
    raise FileNotFoundError("Check the image path.")

#Part A 

img_BGR =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_Gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_HSV =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_LAB =cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_YCrCb =cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

img_RGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(img_RGB)
axes[0].set_title('1. BGR Orginal')
axes[0].axis('off')

axes[1].imshow(img_Gray , cmap='gray')
axes[1].set_title('2.Gray Scale')
axes[1].axis('off')

axes[2].imshow(img_HSV)
axes[2].set_title('3.HSV')
axes[2].axis('off')

axes[3].imshow(img_LAB)
axes[3].set_title('4.LAB')
axes[3].axis('off')

axes[4].imshow(img_YCrCb)
axes[4].set_title('4.YCrCb')
axes[4].axis('off')


plt.tight_layout()
plt.savefig('Color_Difference.png')
plt.show()

H , W = img.shape[:2]

cy , cx = H//2 , W//2

print(f"BGR: {img_BGR[cy, cx]}  "
      f"Grayscale: [{img_Gray[cy, cx]}]  "
      f"HSV: {img_HSV[cy, cx]}  "
      f"LAB: {img_LAB[cy, cx]}  "
      f"YCrCb: {img_YCrCb[cy, cx]}")

lower_orange = np.array([0, 50, 40])
upper_orange = np.array([25, 255, 255])

Mask_HSV = cv2.inRange(img_HSV,lower_orange,upper_orange)

segment_result =cv2.bitwise_and(img, img , mask = Mask_HSV)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')

axes[1].imshow(Mask_HSV, cmap='gray')
axes[1].set_title('Binary Mask')

axes[2].imshow(cv2.cvtColor(segment_result, cv2.COLOR_BGR2RGB))
axes[2].set_title('Segmented Result ')

for ax in axes:
    ax.axis('off')
    
plt.tight_layout()
plt.savefig('hsv_segmentation_result.png')    
plt.show()


# QUESTION 6 ANSWER (HSV Selection Explanation):

# a- Hue Range: Used H from 0 to 255. This wide range covers the entire 
#     color spectrum

#  b- S and V Bounds: S was set from 50 to 255 to filter out low-saturation, 
#     washed-out background colors like the grey asphalt and white sky. 
#     V was set from 40 to 255 to eliminate deep shadows and dark foliage 
#     while keeping the car's well-lit body

#  c- Red Wrap-around: Since H spans from 0 to 255, it already includes both 
#     ends of the red spectrum naturally
