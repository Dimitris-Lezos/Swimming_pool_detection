import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(55, 56):
    try:
        # Load the image
        img = cv2.imread('0000000' + str(i) + '.jpg')
        if not img:
            img = cv2.imread('images/swimmingPool/training/images/0000000'+str(i)+'.jpg')
        # Get it in Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_overlay = img.copy()
        # Convert to HLS to check the color
        img_flt = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        for row in range(img_flt.shape[0]):
            for col in range(img_flt.shape[1]):
                # Check for Cyans
                hue = img_flt[row,col][0]
                lum = img_flt[row,col][1]
                sat = img_flt[row,col][2]
                if (90 < hue and hue < 105) and (50 < lum) and (25 < sat):
                    # Double the intensity of Cyan pixels
                    img_gray[row,col] = min(img_gray[row,col]*2, 255)
                    # For demo put red on the overlay
                    img_overlay[row,col] = [0, 0, 255]
                else:
                    # Half the intensity of pixels with other color
                    img_gray[row,col] = img_gray[row,col]/2
                    pass
        cv2.imshow('Color '+str(i), np.hstack([img, img_overlay]))
        cv2.imshow('Gray '+str(i), np.hstack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_gray]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()

