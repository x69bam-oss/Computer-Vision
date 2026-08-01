import cv2
import numpy  as np
import matplotlib.pyplot as plt

img_bg = cv2.imread('bbg.jpg')
logo = cv2.imread('national_university_logo.jpeg')

if img_bg is None or logo is None:
    raise FileNotFoundError

img_composite = img_bg.copy()

logo = cv2.resize(logo,(200,200))

rows , cols  = logo.shape[:2]

placment = img_composite[0:rows,0:cols]

logo_gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)

_,Mask = cv2.threshold(logo_gray , 240 , 255, cv2.THRESH_BINARY_INV)

Inv_Mask = cv2.bitwise_not(Mask)

img_bg_put = cv2.bitwise_and(placment,placment, mask=Inv_Mask)
logo_put = cv2.bitwise_and(logo,logo,mask=Mask)

Add = cv2.add(img_bg_put,logo_put)
img_composite[0:rows,0:cols] = Add

fig , axes = plt.subplots(1,3,figsize=(15,5))

axes[0].imshow(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
axes[0].set_title('Background')

axes[1].imshow(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB))
axes[1].set_title('Logo')

axes[2].imshow(cv2.cvtColor(img_composite, cv2.COLOR_BGR2RGB))
axes[2].set_title('Final Result')

for ax in axes :
    ax.axis('off')

plt.tight_layout()
plt.savefig("Final Result")
plt.show()