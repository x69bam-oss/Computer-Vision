import cv2
import numpy as np
import matplotlib.pyplot as plt

#lood the Image 

img = cv2.imread('1.jpeg')
img2 = cv2.imread('2.jpeg')

if img is None :
    raise FileNotFoundError("Check the image path.")

H , W = img.shape[:2]

croped_img = img[H//4 : H//2 , W//3 : (W*2)//3 ]

print("Orginal Image " , img.shape)
print("Cropped Image " , croped_img.shape)



Hc , Wc = croped_img.shape[:2]

half_Crop = cv2.resize(croped_img, (Wc//2 , Hc//2) ,interpolation=cv2.INTER_AREA )
doubl_Crop= cv2.resize(croped_img , (Wc*2 , Hc*2), interpolation=cv2.INTER_CUBIC)
fixed_Crop= cv2.resize(croped_img,(200,200),interpolation=cv2.INTER_LINEAR)

fiig , axes =plt.subplots(1,3,figsize=(15,5))

axes[0].imshow(cv2.cvtColor(half_Crop,cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Half Cropped\n{half_Crop.shape[:2]}')


axes[1].imshow(cv2.cvtColor(doubl_Crop,cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Double Cropped\n{doubl_Crop.shape[:2]}')


axes[2].imshow(cv2.cvtColor(fixed_Crop,cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Fixed Cropped\n{fixed_Crop.shape[:2]}')

for ax in axes :
    ax.axis('off')

plt.tight_layout()
plt.savefig('step2_cropped.png')
plt.show()

Angle_1 = 30
Angle_2 = -30

Center = (H//2 , W//2)

Img_Rot_1 = cv2.getRotationMatrix2D(Center,Angle_1,1.0)
Rotated_30 = cv2.warpAffine(img,Img_Rot_1,(H,W))

Img_Rot_2 = cv2.getRotationMatrix2D(Center,Angle_2,1.0)
Rotated_Miuns_30 = cv2.warpAffine(img,Img_Rot_2,(H,W))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(Rotated_30, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Rotated +{Angle_1}°')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(Rotated_Miuns_30, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Rotated {Angle_2}° (Clockwise)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('step3_rotated.png')
plt.show()
 
flip_vertical = cv2.flip(img, 0)
flip_horizontal = cv2.flip(img, 1)
flip_both = cv2.flip(img, -1)

fig , axes = plt.subplots(1, 3, figsize=(15, 5))
images = [flip_vertical, flip_horizontal, flip_both]
titles = ['Vertical Flip', 'Horizontal Flip', 'Both Flips']
for ax, image, title in zip(axes, images, titles):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
    
plt.tight_layout()
plt.savefig('step4_flipped.png')
plt.show()

img2_resized = cv2.resize(img2, (W, H))

Alpha_1 = 0.25
Alpha_2 = 0.50
Alpha_3 = 0.75

blinded_1 = cv2.addWeighted(img, Alpha_1, img2_resized, 1 - Alpha_1, 0)
blinded_2 = cv2.addWeighted(img, Alpha_2, img2_resized, 1 - Alpha_2, 0)
blinded_3 = cv2.addWeighted(img, Alpha_3, img2_resized, 1 - Alpha_3, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

blends = [blinded_1, blinded_2, blinded_3]
blend_titles = [f'Alpha = {Alpha_1}', f'Alpha = {Alpha_2}', f'Alpha = {Alpha_3}']

for ax, b_img, title in zip(axes, blends, blend_titles):
    ax.imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('step5_blended.png')
plt.show()

RECT_X1, RECT_Y1 = 50, 50
RECT_X2, RECT_Y2 = 250, 200
CIRCLE_CENTER = (400, 300)
CIRCLE_RADIUS = 80

mask = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(mask, (RECT_X1, RECT_Y1), (RECT_X2, RECT_Y2), 255, -1)
cv2.circle(mask, CIRCLE_CENTER, CIRCLE_RADIUS, 255, -1)

mask_inv = cv2.bitwise_not(mask)

masked_img = cv2.bitwise_and(img, img, mask=mask)
masked_img_inv = cv2.bitwise_and(img, img, mask=mask_inv)

fig,axes = plt.subplots(1,2,figsize=(15,5))
axes[0].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Masked Image (Rectangle + Circle)')

axes[1].imshow(cv2.cvtColor(masked_img_inv, cv2.COLOR_BGR2RGB))
axes[1].set_title('Masked Image (Inverse)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('step6_masked.png')
plt.show()