import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from PIL import Image
import io
import codecs

# def get_image_names_list():
#     return os.listdir('Data/')

# def create_image_path(image_name):
#     return 'Data/'+image_name

# def read_image(image_full_path):
#     return cv2.imread(image_full_path)

# def convert_base64_string_to_image_numpy_array(image_data):
#     image_data = image_data.replace("data:image/jpeg;base64,","")
#     image_data_bytes = image_data.encode('utf-8')
#     decoded_image_data = codecs.decode(image_data_bytes, 'utf-8')
#     image = Image.open(decoded_image_data)
#     image_array = np.array(image)   

#     return image_array

# def convert_image_numpy_array_to_string(image_data):
#     # Convert the NumPy array to a base64-encoded string
#     image_string = np.base64.b64encode(image_data).decode('utf-8')

#     return image_string

def plot_image_hist(image):
    range1 = 5
    range2 = 256
    bins = range2 - range1 
    plt.hist(image.ravel(),
            bins=bins,
            range=[range1,range2],
            fc='k',
            ec='k') 

    plt.show()

def decode_image(image_data):
    image_data = image_data.replace("data:image/jpeg;base64,","")
    # Decode the base64-encoded image content
    image_data = base64.b64decode(image_data)
    # Decode the byte stream into a NumPy array
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    return image

def encode_image(image):
    # Encode the image object to a byte stream
    encoded_image = cv2.imencode('.png', image)[1].tostring()
    #print(encoded_image)
    # Convert the byte stream to a Base64-encoded string
    image = base64.b64encode(encoded_image).decode('utf-8')
    #print(base64_encoded_string)
    #image = "data:image/jpeg;base64," + base64_encoded_string
    #print(type(image))
    #print(image)
    return image

    # Convert the base64 string to a format that can be used in the Dash app
    encoded_image = str(encoded_image.tobytes())

    return encoded_image

def draw_contours(image, contours):
    window_name = 'Pool Satelite Photo'
    cv2.drawContours(image, contours, -1, (150, 150, 255), 2)
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def show_image_with_waitkey(image):
    window_name = 'Pool Satelite Photo'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_pools(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
    return image, thresholded_image, contours

def pool_detection(content):
    image = decode_image(content)
    print(image)
    image, thresholded_image, contours = find_pools(image)
    image = draw_contours(image, contours)
    encoded_image = encode_image(image) 
    print(encoded_image)
    return encode_image