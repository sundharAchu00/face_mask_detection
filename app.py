import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model("face_mask_detector_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Define labels
labels = ["with_mask", "without_mask", "mask_weared_incorrect"]

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 128x128 (the input size of the model)
    resized_frame = cv2.resize(frame, (128, 128))

    # Normalize the image (same as during training)
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to match the input shape (batch_size, height, width, channels)
    expanded_frame = np.expand_dims(normalized_frame, axis=0)

    # Make prediction
    prediction = model.predict(expanded_frame)

    # Get predicted class
    predicted_class = np.argmax(prediction)

    # Display the result on the frame
    cv2.putText(
        frame,
        f"Prediction: {labels[predicted_class]}, Confidence: {float(np.max(prediction))}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Show the webcam feed
    cv2.imshow("Webcam - Face Mask Detection", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
