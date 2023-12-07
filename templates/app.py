from dash import Dash, dcc, html, Input, Output
import numpy as np
import base64
import cv2
import numpy as np
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
    if contents is not None:
        # Decode image data
        data = contents.split(',')[1]
        encoded_image = pool_detection(data)
        data = data.replace("data:image/jpeg;base64,","")
        decoded = base64.b64decode(data)

        # Convert decoded data to NumPy array
        image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = (90, 50, 50)
        upper_blue = (130, 255, 255)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(image, image, mask=mask)
        grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(grey, 98, 100, cv2.THRESH_BINARY)
        # Find contours in the image
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        processed_image = cv2.drawContours(image, contours, -1, (150, 150, 255), 2)

        # Convert processed image to base64 encoding
        encoded_image = cv2.imencode('.png', processed_image)[1].tostring()
        encoded_image = base64.b64encode(encoded_image).decode('utf-8')

        # Update the output image element with the encoded image data
        output_image_element = html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        return output_image_element

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)