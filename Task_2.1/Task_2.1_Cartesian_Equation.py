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
   
   for j in range(cx-r,cx+r+1)  :
       y_sqrt= r**2 -(j-cx)**2
       if y_sqrt>=0:
           y_off_set = math.sqrt(y_sqrt)
           y1=int(cy+y_off_set)
           y2=int(cy-y_off_set)
           img[j,y1]=150
           img[j,y2]=150
       

img = cv2.imshow('Image', img)

img = cv2.waitKey(0)
img = cv2.destroyAllWindows()
