import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

input_dimensions = (64, 64)

app = Flask(__name__)

model = load_model("braintumor10epochs.h5")
print("Model loaded successfully.")

def class_number_to_result(class_number):
    if class_number == 0:
        return "Tumor not present."
    else:
        return "Tumor present."

def get_result(image_path):
    image = cv2.imread(image_path)
    image = Image.fromarray(image, "RGB")
    image = image.resize(input_dimensions)
    image = np.array(image)
    input_image = np.expand_dims(image, axis=0)
    result = model.predict(input_image)
    return result[0][0]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "uploads", secure_filename(file.filename))
        file.save(file_path)
        value = get_result(file_path)
        result = class_number_to_result(value)
        return result

if __name__ == "__main__":
    app.run(debug=True)