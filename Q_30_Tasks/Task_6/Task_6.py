import cv2
import numpy  as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')

if img is None :
    raise FileNotFoundError

Img_Modify = img.copy()

H , W = img.shape[:2]

Img_Modify[0:40 , 0 :40] = [0 , 0, 255] #For Red square

Img_Modify[H//2 , :]=[255,255,255] #For White Hor_Line

Img_Modify[ : , W//2]=[255,255,255] #For White Ver_line

#Now For The Green Trangle

X = (W//2) + 30
Y = (H//2) + 30

Img_Modify[Y : Y+100 , X : X+100] = [0,255,0] 

#Displaying The Result

fig , axes = plt.subplots(1,2,figsize=(8,8))

axes[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0].set_title('Oringal Image')

axes[1].imshow(cv2.cvtColor(Img_Modify,cv2.COLOR_BGR2RGB))
axes[1].set_title('Modified Image')

for ax in axes :
    ax.axis('off')

plt.tight_layout()
plt.savefig("Difference")
plt.show()    