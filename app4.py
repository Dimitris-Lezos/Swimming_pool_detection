from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import datetime
import base64
import cv2
import numpy as np
from scripts.utils import pool_detection

app = Dash(__name__)


app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(content, filename, date):
    # Decode image data
    data = content.split(',')[1]
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

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
# HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src= output_image_element),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(output_image_element[0:200] + '...'
                 , style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified')
)
def update_output(contents, list_of_names, list_of_dates):
    if contents is not None: 
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)