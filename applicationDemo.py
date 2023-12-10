from dash import Dash, dcc, html, Input, Output
import numpy as np
import base64
import cv2
from scripts.locator import locateArea


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
        #encoded_image = pool_detection(data)
        data = data.replace("data:image/jpeg;base64,","")
        decoded = base64.b64decode(data)

        # Convert decoded data to NumPy array
        input_image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
        img_overlay, image = locateArea(input_image)

        # Convert processed image to base64 encoding
        encoded_image = cv2.imencode('.png', image)[1].tostring()
        encoded_image = base64.b64encode(encoded_image).decode('utf-8')

        # Update the output image element with the encoded image data
        output_image_element = html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        # Convert input image to base64 encoding
        encoded_image = cv2.imencode('.png', input_image)[1].tostring()
        encoded_image = base64.b64encode(encoded_image).decode('utf-8')
        input_image_element = html.Img(src='data:image/png;base64,{}'.format(encoded_image))

        # return output_image_element
#        return html.Div(html.Table(html.Tr([html.Td(html.Div(html.Img(data))),html.Td('==>'),html.Td(output_image_element)])))
        return html.Div(html.Table(html.Tr([html.Td(input_image_element),html.Td('==>'),html.Td(output_image_element)])))
        # return html.Div([output_image_element, html.Table(html.) ,output_image_element])

if __name__ == '__main__':
    app.run_server(host = "0.0.0.0") #, port="8080", debug=True, use_reloader=False # http://192.168.1.7:8080/