import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Assuming TensorFlow 2.x

# Load the pre-trained emotion classification model
model = load_model('model_weights.h5')

# Initialize video capture object
video = cv2.VideoCapture(0)

# Load the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels dictionary
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                   4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi = gray[y:y + h, x:x + w]

        # Resize the ROI to match the model's input size
        resized_roi = cv2.resize(roi, (64, 64))

        # Normalize the pixel values (typically between 0 and 1)
        normalized_roi = resized_roi / 255.0

        # Reshape the normalized ROI for model prediction
        reshaped_roi = np.expand_dims(normalized_roi, axis=0)  # Add batch dimension

        # Make prediction using the model
        prediction = model.predict(reshaped_roi)

        # Get the predicted emotion label (index of maximum probability)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the predicted emotion text above the face
        cv2.putText(frame, emotion_labels[predicted_label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Facial Emotion Recognition', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release video capture object and destroy all windows
video.release()
cv2.destroyAllWindows()