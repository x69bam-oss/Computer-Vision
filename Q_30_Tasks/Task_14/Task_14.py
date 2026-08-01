import cv2
import numpy  as np
import matplotlib.pyplot as plt

def Show_All_Spaces(path) :

    img = cv2.imread(path)

    if img is None :
        raise FileNotFoundError("Please Check The Image Path {path}")

    H , W = img.shape[:2]

    cy, cx = H//2 , W//2

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    print(f"BGR : {img[cx,cy]}"   )
    print(f"Gray : {img_gray[cx,cy]}")
    print(f"HSV : {img_hsv[cx,cy]}")
    print(f"LAB : {img_lab[cx,cy]}")
    print(f"YCrCb : {img_ycrcb[cx,cy]}")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('BGR (Original)')
    
    
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title('Gray ')
    
    axes[2].imshow(img_hsv)
    axes[2].set_title('HSV')
    
    axes[3].imshow(img_lab)
    axes[3].set_title('LAB')
    
    axes[4].imshow(img_ycrcb)
    axes[4].set_title('YCrCb')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("Result.png")
    plt.show()

    
Show_All_Spaces(r"X:\Computer Vision\Q_30_Tasks\Red_Apple.jpg")