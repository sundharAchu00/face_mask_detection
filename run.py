import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained face mask detection model
model = load_model("face_mask_detector_model_v1.h5")

# Define the labels
labels = ["with_mask", "without_mask", "mask_weared_incorrect"]


# Image preprocessing function
def preprocess_image(image_bytes):
    # Convert byte data to an image
    img = Image.open(io.BytesIO(image_bytes))

    # Resize the image to match the input shape expected by the model
    img = img.resize((128, 128))

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Add batch dimension (since model expects input in (batch, height, width, channels) format)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Route to handle mask detection
@app.route("/detect_mask", methods=["POST"])
def detect_mask():
    try:
        # Get the image from the request
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read the image bytes
        img_bytes = file.read()

        # Preprocess the image
        img_array = preprocess_image(img_bytes)

        # Make prediction using the model
        prediction = model.predict(img_array)

        print(prediction)

        # Get the predicted class (with_mask, without_mask, mask_weared_incorrect)
        predicted_class = np.argmax(prediction)

        print(predicted_class)

        # Return the result as a JSON response
        result = {
            "prediction": labels[predicted_class],
            "confidence": float(np.max(prediction)),  # Confidence of prediction
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
