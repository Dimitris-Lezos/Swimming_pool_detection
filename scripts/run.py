import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_image_hist, show_image_with_waitkey2, show_image_with_waitkey


images_path = 'Data/'
image_name = '000000148.jpg'
image_full_path = images_path+image_name

window_name = 'Pool Satelite Photo'
image = cv2.imread(image_full_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = (90, 50, 50)
upper_blue = (130, 255, 255)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(image, image, mask=mask)
grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# Threshold the image to keep only pixels greater than 98 and lower than 100
_, thresholded_image = cv2.threshold(grey, 98, 100, cv2.THRESH_BINARY)
# Find the contours of the non-zero pixels
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#plot_image_hist(image)
show_image_with_waitkey(thresholded_image, window_name)
show_image_with_waitkey2(image, window_name, contours)