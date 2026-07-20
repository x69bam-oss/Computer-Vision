#Importing Needed Libraries

import cv2
import numpy as np  
import matplotlib.pyplot as plt

# Reading the images

img_1 = cv2.imread('1.jpeg')
img_2 = cv2.imread('2.jpeg')
img_3 = cv2.imread('3.jpeg')

#Image Checking 

if img_1  is None or img_2 is None or img_3 is None :
    raise FileNotFoundError("One or more images not found. Please check the file paths.")

#_______________________________________________________________________________

#Image structure ;3

#Displaying the shape of The Images

print ("Image_1 Shape :" , img_1.shape)
print ("Image_2 Shape :" , img_2.shape)
print ("Image_3 Shape :" , img_3.shape)

#Displaying Data Type of The Images 

print ("Image_1 dType :" , img_1.dtype)
print ("Image_2 dType :" , img_2.dtype)
print ("Image_3 dType :" , img_3.dtype)

#Pixels Count For the Images = height * width

print ("Image_1 Tot_Pixel :" , img_1.shape[0]*img_1.shape[1]*img_1.shape[2])
print ("Image_2 Tot_Pixel :" , img_2.shape[0]*img_2.shape[1]*img_2.shape[2])
print ("Image_3 Tot_Pixel :" , img_3.shape[0]*img_3.shape[1]*img_3.shape[2])

#_______________________________________________________________________________

#Center Pixel Accessing For the Images

h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
h3, w3 = img_3.shape[:2]

cx1 ,cy1 = w1//2 , h1//2
cx2 ,cy2 = w2//2 , h2//2   
cx3 ,cy3 = w3//2 , h3//2

print ("B1_values :" , img_1[cy1,cx1,0])
print ("G1_values :" , img_1[cy1,cx1,1])
print ("R1_values :" , img_1[cy1,cx1,2])



print ("B2_values :" , img_2[cy2,cx2,0])
print ("G2_values :" , img_2[cy2,cx2,1])
print ("R2_values :" , img_2[cy2,cx2,2])



print ("B3_values :" , img_3[cy3,cx3,0])
print ("G3_values :" , img_3[cy3,cx3,1])
print ("R3_values :" , img_3[cy3,cx3,2])

#_______________________________________________________________________________

#Drawing the circle in the Images

r1 = min(h1,w1)//10
r2 = min(h2,w2)//10
r3 = min(h3,w3)//10

y1 ,x1 = np.ogrid[:h1 ,:w1]
y2 ,x2 = np.ogrid[:h2 ,:w2]
y3 ,x3 = np.ogrid[:h3 ,:w3]

Mask_1 = (x1-cx1)**2 + (y1-cy1)**2 <= r1**2
Mask_2 = (x2-cx2)**2 + (y2-cy2)**2 <= r2**2
Mask_3 = (x3-cx3)**2 + (y3-cy3)**2 <= r3**2

img_1[Mask_1] =[255 , 255, 255]
img_2[Mask_2] =[255 , 255, 255]
img_3[Mask_3] =[255 , 255, 255]

#_______________________________________________________________________________

#Displaying the images using matplotlib

B1 ,G1 ,R1 = img_1[:,:,0], img_1[:,:,1], img_1[:,:,2] #Splitting the channel usuing slicing
B2 ,G2 ,R2 = img_2[:,:,0], img_2[:,:,1], img_2[:,:,2] 
B3 ,G3 ,R3 = img_3[:,:,0], img_3[:,:,1], img_3[:,:,2]

fig , axes =plt.subplots(1,3,figsize=(12,12)) #Setting the subplot grid to 1x3 and size to 12x12

for ax, channel, color, img_num in zip(axes.flatten(),  #For (Zip) loop with 4 variables to iterate over the axes, channels, colors, and image numbers
                                       [B1,G1,R1],
                                       ['Blue','Green','Red'],
                                       ['Image 1'] + ['Image 2'] + ['Image 3']):
    ax.imshow(channel , cmap='gray') #we used gray cmap cuz it's a 2D matrix and we want to display it in grayscale 
    ax.set_title(f'{img_num} {color} Channel') #Puts a title on each subplot with the image number and channel color
    ax.axis('off') #Removes the axis ticks and labels from each subplot to make the images somther
plt.tight_layout() #Makes sure the subplots do not overlap in each other
plt.savefig('channel_split_1.png')  #Saving the figure as a PNG file
plt.show()

fig , axes =plt.subplots(1,3,figsize=(12,12))

for ax, channel, color, img_num in zip(axes.flatten(),
                                       [B2,G2,R2],
                                       ['Blue','Green','Red'],
                                       ['Image 1'] + ['Image 2'] + ['Image 3']):
    ax.imshow(channel , cmap='gray')
    ax.set_title(f'{img_num} {color} Channel') 
    ax.axis('off')
plt.tight_layout()
plt.savefig('channel_split_2.png')  
plt.show()

fig , axes =plt.subplots(1,3,figsize=(12,12))

for ax, channel, color, img_num in zip(axes.flatten(),
                                       [B3,G3,R3],
                                       ['Blue','Green','Red'],
                                       ['Image 1'] + ['Image 2'] + ['Image 3']):
    ax.imshow(channel , cmap='gray')
    ax.set_title(f'{img_num} {color} Channel') 
    ax.axis('off')
plt.tight_layout()
plt.savefig('channel_split_3.png')  
plt.show()

#_______________________________________________________________________________

# float32 Conversion 0_o

img_f_1 = img_1.astype(np.float32)
img_f_2 = img_2.astype(np.float32)
img_f_3 = img_3.astype(np.float32)

print ("Image 1 float center BGR values:",img_f_1[cy1,cx1])
print ("Image 2 float center BGR values:",img_f_2[cy2,cx2])
print ("Image 3 float center BGR values:",img_f_3[cy3,cx3])

X = int(0)

result_f_1 = img_f_1 + (int(X)/255.0)
result_f_2 = img_f_2 + (int(X)/255.0)
result_f_3 = img_f_3 + (int(X)/255.0)

result_f_1 = np.clip(result_f_1, 0.0, 1.0)  #Clipping the values to [0.0,1.0]
result_f_2 = np.clip(result_f_2, 0.0, 1.0)
result_f_3 = np.clip(result_f_3, 0.0, 1.0)

back_to_uint8_1 = (result_f_1 * 255).astype(np.uint8)  #Converting back to uint8
back_to_uint8_2 = (result_f_2 * 255).astype(np.uint8)
back_to_uint8_3 = (result_f_3 * 255).astype(np.uint8)

print("back_to_uint8_1 value :" , back_to_uint8_1[cx1,cy1])
print("back_to_uint8_2 value :" , back_to_uint8_2[cx2,cy2])
print("back_to_uint8_3 value :" , back_to_uint8_3[cx3,cy3])


cv2.imwrite('output_1.jpg', img_1)
cv2.imwrite('output_2.jpg', img_2)
cv2.imwrite('output_3.jpg', img_3)