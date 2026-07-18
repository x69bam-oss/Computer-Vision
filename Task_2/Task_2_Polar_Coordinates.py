import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

x_index = 500
y_index = 500

img = np.zeros(x_index * y_index, dtype=np.uint8).reshape(x_index, y_index)

for i in range(50 ,300):
   img[i,i]=150
   
   cx ,cy = 100 ,100
   r=50
   
#using the Polar Coordinates is more efficent cuz it will remove the blank spots in Acute Angles
#Polar Coordinates Equations is :

    # X = Cx + r * Cos(theata)
    # Y = Cy + r * sin(theata)
        
num_segments = 360 

for i in range(num_segments):
    theta = (2 * math.pi * i) / num_segments
    x = int(cx + r * math.cos(theta))
    y = int(cy + r * math.sin(theta))
   
    #The Stantment to make sure that the pixcl we will draw is in the borders of the image
    
    if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]: 
        img[x, y] = 150
       

img = cv2.imshow('Image', img)

img = cv2.waitKey(0)
img = cv2.destroyAllWindows()
