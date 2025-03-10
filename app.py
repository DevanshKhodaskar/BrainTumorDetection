from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]

        # Read image into memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO segmentation
        results = model(img)
        output_img = results[0].plot()  # Processed image

        # Convert image to bytes
        _, buffer = cv2.imencode(".jpg", output_img)
        image_io = BytesIO(buffer)

        return send_file(image_io, mimetype="image/jpeg")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
