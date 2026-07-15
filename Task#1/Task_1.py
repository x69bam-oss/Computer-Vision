#Importing Needed Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the images

img_1 =cv2.imread('1.jpg')  
img_2 =cv2.imread('2.jpg')
img_3 =cv2.imread('3.png')

#Image Checking 

if img_1 is None or img_2 is None or img_3 is None:
    raise FileNotFoundError("One or more images not found. Please check the file paths.")

#_______________________________________________________________________________

#Image structure ;3

#Displaying the shape of The Images

print("Image 1 Shape :" ,img_1.shape)
print("Image 2 Shape :" ,img_2.shape)   
print("Image 3 Shape :" ,img_3.shape)

#Displaying Data Type of The Images 

print("Image 1 Data Type :" ,img_1.dtype)
print("Image 2 Data Type :" ,img_2.dtype)
print("Image 3 Data Type :" ,img_3.dtype)

#Pixels Count For the Images = height * width

print("Image 1 Pixels :" ,img_1.shape[0]*img_1.shape[1])
print("Image 2 Pixels :" ,img_2.shape[0]*img_2.shape[1])
print("Image 3 Pixels :" ,img_3.shape[0]*img_3.shape[1])

#_______________________________________________________________________________

#Center Pixel Accessing For the Images

h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
h3, w3 = img_3.shape[:2]

cx1 ,cy1 = w1//2 , h1//2
cx2 ,cy2 = w2//2 , h2//2   
cx3 ,cy3 = w3//2 , h3//2

print ("Image 1 Center BGR values:",img_1[cy1,cx1])
print ("Image 2 Center BGR values:",img_2[cy2,cx2])
print ("Image 3 Center BGR values:",img_3[cy3,cx3])

#_______________________________________________________________________________

#Displaying the images using matplotlib

rgb_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)  #Converting BGR to RGB
rgb_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2RGB)
rgb_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,8))
plt.subplot(1,3,1) #Arranging the images in a 1x3 grid
plt.imshow(rgb_1)
plt.title(f'Image 1 Shape : {img_1.shape} , dType : {img_1.dtype}')
plt.axis('off') #Removes the axis ticks and labels from each subplot to make the images smother

plt.subplot(1,3,2)
plt.imshow(rgb_2)
plt.title(f'Image 2 Shape : {img_2.shape} , dType : {img_2.dtype}')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(rgb_3)
plt.title(f'Image 3 Shape : {img_3.shape} , dType : {img_3.dtype}')
plt.axis('off')

plt.tight_layout() #Makes sure the subplots do not overlap in each other
plt.show()

#_______________________________________________________________________________

#Split Channels & float32 Conversion 0_o

#First Float32 Conversion 0_o

img_f_1 = img_1.astype(np.float32)  #Converting to float32
img_f_2 = img_2.astype(np.float32)
img_f_3 = img_3.astype(np.float32)

print ("Image 1 float center BGR values:",img_f_1[cy1,cx1])
print ("Image 2 float center BGR values:",img_f_2[cy2,cx2])
print ("Image 3 float center BGR values:",img_f_3[cy3,cx3])

X = input("Enter the value to be added to the center pixel for all images : ") #Prompting the user to enter a value

result_f_1 = img_f_1 + (int(X)/255.0)
result_f_2 = img_f_2 + (int(X)/255.0)
result_f_3 = img_f_3 + (int(X)/255.0)

result_f_1 = np.clip(result_f_1, 0.0, 1.0)  #Clipping the values to [0.0,1.0]
result_f_2 = np.clip(result_f_2, 0.0, 1.0)
result_f_3 = np.clip(result_f_3, 0.0, 1.0)

back_to_uint8_1 = (result_f_1 * 255).astype(np.uint8)  #Converting back to uint8
back_to_uint8_2 = (result_f_2 * 255).astype(np.uint8)
back_to_uint8_3 = (result_f_3 * 255).astype(np.uint8)

cv2.imwrite('result_image_1.png', back_to_uint8_1)  #Saving the result images
cv2.imwrite('result_image_2.png', back_to_uint8_2)
cv2.imwrite('result_image_3.png', back_to_uint8_3)

#___________________________________________________________________________________

#Second Split T-T

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