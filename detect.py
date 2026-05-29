import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/emotion_model.h5')

# Emotion labels
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# Load Haar Cascade
face_classifier = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Extract face
        roi_gray = gray[y:y+h, x:x+w]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except:
            continue

        # Normalize
        roi = roi_gray.astype('float32') / 255.0

        # Reshape
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Predict
        prediction = model.predict(roi, verbose=0)[0]

        print("Prediction:", prediction)

        # Get highest probability
        max_index = np.argmax(prediction)

        confidence = prediction[max_index]

        label = emotion_labels[max_index]

        # Show only if confidence is enough
        if confidence > 0.2:

            text = f"{label} ({confidence:.2f})"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()