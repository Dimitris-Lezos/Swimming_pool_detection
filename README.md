# Swimming_pool_detection

Here is a summary of the readme file:

# Machine Vision Pool Detection
This is a web application that uses Python Dash to detect pools in images. The application has a simple user interface with an upload button for selecting an image. Once an image is uploaded, the application will process the image and display the results, including the detected pools.

# Prerequisites
To run this application, you will need the following:

- Python 3.6 or higher
- Dash
- OpenCV
- NumPy
- base64

# Installation
To install the required dependencies, you can use pip:

``` Bash
pip install dash opencv-python numpy base64
```

# Usage
To run the application, you can start the Dash server:

```Bash
python app.py
```

Then, open a web browser and go to ```http://localhost:8050```. You should see the application's user interface. Upload an image to the application and it will process the image and display the results.

# How it works
The application uses the OpenCV library to detect pools in images. The application first converts the image to grayscale, then applies a threshold to the image, and finally finds contours in the image. The contours are then used to draw the detected pools on the image.
