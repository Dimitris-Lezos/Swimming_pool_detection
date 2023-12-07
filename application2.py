from dash import Dash, dcc, html, Input, Output
import numpy as np
import base64
import cv2
from scripts.utils import pool_detection
import matplotlib.pyplot as plt


application = Dash(__name__)

app = application

app.layout = html.Div(
    children=[
        html.H1(
            children="Machine Vision Pool Detection",
            style={
                "textAlign": "center",
                "color": "#2980b9",
                "font-size": "24px",
                "font-weight": "bold",
                "margin": "20px 0",
            },
        ),
        dcc.Upload(
            id="upload-image",
            children=[
                html.Div(
                    children=[
                        html.Span(
                            children="Drag and Drop or",
                            style={
                                "color": "#6c757d",
                                "fontSize": "14px",
                                "marginRight": "10px",
                            },
                        ),
                        html.A(
                            children="Select One Image",
                            href="#",
                            style={
                                "color": "#007bff",
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "textDecoration": "none",
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "padding": "10px",
                        "backgroundColor": "#fff",
                        "boxShadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
                        "border-color": "#2980b9",
                    },
                )
            ],
            multiple=False,
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "margin": "20px 0",
            },
        ),
        html.Div(id="output-image", style={"textAlign": "center", "margin": "20px 0"}),
    ]
)

@app.callback(
    Output('output-image', 'children'),
    Input('upload-image', 'contents')
)



def update_output(contents):
    def dilateConnected(img_mask, img_extended_mask, img_overlay=None, color=[0,255.0]): #-> cv2.typing.MatLike:
        # Dilate selection if in additional mask (expected hue)
        # img_dilate = img_mask.copy()
        direction = 0
        while (True):
            continue_dilation = False
            row_range = [
                range(1, img_mask.shape[0] - 1),  # top-bottom to left-right
                range(1, img_mask.shape[0] - 1),  # top-bottom to right-left
                range(img_mask.shape[0] - 2, 1, -1),  # bottom-top to right-left
                range(img_mask.shape[0] - 2, 1, -1),  # bottom-top to left-right
            ]
            col_range = [
                range(1, img_mask.shape[1] - 1),  # top-bottom to left-right
                range(img_mask.shape[1] - 2, 1, -1),  # top-bottom to right-left
                range(img_mask.shape[1] - 2, 1, -1),  # bottom-top to right-left
                range(1, img_mask.shape[1] - 1),  # bottom-top to left-right
            ]
            for row in row_range[direction % 4]:
                for col in col_range[direction % 4]:
                    # If the pixel is not marked in initial range but was marked from the extended range operation
                    if img_mask[row, col] == 0 and img_extended_mask[row, col] > 0:
                        # If an adjacent pixel is marked then mark this one also
                        for i in range(3):
                            for j in range(3):
                                if img_mask[row + i - 1, col + j - 1] > 0:
                                    continue_dilation = True
                                    img_mask[row, col] = 255
                                    img_overlay[row, col] = color
            print(".", end="")
            if continue_dilation == False:
                print("Stopping")
                break
            direction += 1
        return img_mask
    
    if contents is not None:
        # Decode image data
        data = contents.split(',')[1]
        encoded_image = pool_detection(data)
        data = data.replace("data:image/jpeg;base64,","")
        decoded = base64.b64decode(data)

        # Convert decoded data to NumPy array
        image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
        img_overlay = image.copy()
        # Should we blur first?
        # Convert to HLS to check the color
        img_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        # Select the pixels in Range to the expected colors
        # This is the initial Range
        img_range = cv2.inRange(img_hls, (90, 75, 100), (105, 255, 255))
        # This is the extended Range (some false identifications here)
        img_extend_range = cv2.inRange(img_hls, (90, 50, 50), (105, 255, 255))
        # Keep the extended range only if it connects to the initial range
        img_range = dilateConnected(img_range, img_extend_range, img_overlay, [0, 255, 0])
        # Create an even more extended range (many false identifications here)
        img_extend_range = cv2.inRange(img_hls, (75, 50, 25), (105, 255, 255))
        ############################################################################
        # Color Initial Range pixels with Red and Extended with Yellow to show our work
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_extend_range[row,col] > 0:
                    img_overlay[row,col] = [0,255,255]
                if img_range[row,col] > 0:
                    img_overlay[row,col] = [0,0,255]
        img_mask = img_range.copy()
        # Do some erosion to remove small areas
        for _ in range(3):
            img_mask = cv2.erode(img_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        # Run connected components and keep only "big" components
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(img_mask, 4, cv2.CV_32S)
        for row in range(img_mask.shape[0]):
            for col in range(img_mask.shape[1]):
                id = label_ids[row,col]
                if id > 0 and values[id][4] < 35:
                    img_mask[row,col] = 0
        # Remove eroded pixels from img_overlay to show our work (make them green)
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_range[row,col] != img_mask[row,col]:
                    img_overlay[row,col] = [0, 255, 0]
        # Mark pool pixels in img_overlay to show our work (make them red)
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_mask[row,col] > 0:
                    img_overlay[row,col] = [0, 0, 255]
        # Dilate selection if in extended range mask (expected hue)
        img_mask = dilateConnected(img_mask, img_extend_range, img_overlay, [255, 0, 0])
        # Finally Dilate and Erode to get rid of holes
        for _ in range(4):
            img_mask = cv2.dilate(img_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        for _ in range(4):
            img_mask = cv2.erode(img_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        img_overlay_dil = img_overlay.copy()
        # Mark in overlay image with Blue to show our work
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_mask[row,col] > 0:
                    img_overlay_dil[row,col] = [255, 0, 0]
        ##########################################################
        # Draw Contours over initial image and show results
        ret, threshold = cv2.threshold(img_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(image, contours, -1, (255, 0, 255), 2)

        # Convert processed image to base64 encoding
        encoded_image = cv2.imencode('.png', image)[1].tostring()
        encoded_image = base64.b64encode(encoded_image).decode('utf-8')

        # Update the output image element with the encoded image data
        output_image_element = html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        return output_image_element

if __name__ == '__main__':
    app.run_server(host = "0.0.0.0") #, port="8080", debug=True, use_reloader=False # http://192.168.1.7:8080/