import cv2
import numpy as np

img = cv2.imread('B3.jpg')

if img is None :
    raise FileNotFoundError

img_B = img 

#Convert BGR to HSV

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
 
 #Define The Colour we are looking for 
 
lower_blue = np.array([100,50,50])
Upper_blue = np.array([150,255,255])
 
mask = cv2.inRange(hsv , lower_blue , Upper_blue)
mask_inve = cv2.bitwise_not(mask)
 
lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
 
L , A , B = cv2.split(lab)
 
L_lab = L
 
NL_lab=cv2.add(L_lab,40)

lab_ench = cv2.merge([NL_lab,A,B])

result_lab = cv2.cvtColor(lab_ench,cv2.COLOR_Lab2BGR)



img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Gray_BGR = cv2.cvtColor(img_Gray,cv2.COLOR_GRAY2BGR)

For_Ground = cv2.bitwise_and(result_lab,result_lab,mask=mask)
Back_Ground = cv2.bitwise_and(Gray_BGR , Gray_BGR , mask = mask_inve)

Final_Image = cv2.add(For_Ground,Back_Ground)
 
 
 
 
 
 
result = cv2.bitwise_and(img, img , mask = mask)

cv2.imshow("Image :",img)
cv2.imshow("Final Image :" ,Final_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()

