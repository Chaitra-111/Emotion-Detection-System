import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Load trained model
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

# ==========================
# Emotion Prediction Function
# ==========================
def detect_emotion(face_gray):

    roi_gray = cv2.resize(face_gray, (48, 48))

    roi = roi_gray.astype('float32') / 255.0

    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    prediction = model.predict(roi, verbose=0)[0]

    max_index = np.argmax(prediction)

    confidence = prediction[max_index]

    emotion = emotion_labels[max_index]

    return emotion, confidence

# ==========================
# Upload Image Function
# ==========================
def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]

        emotion, confidence = detect_emotion(roi_gray)

        text = f"{emotion} ({confidence:.2f})"

        cv2.putText(
            image,
            text,
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    # Convert image for GUI display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image_rgb)

    pil_image = pil_image.resize((700, 500))

    tk_image = ImageTk.PhotoImage(pil_image)

    image_label.config(image=tk_image)

    image_label.image = tk_image

# ==========================
# Webcam Function
# ==========================
def open_webcam():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            roi_gray = gray[y:y+h, x:x+w]

            emotion, confidence = detect_emotion(roi_gray)

            text = f"{emotion} ({confidence:.2f})"

            cv2.putText(
                frame,
                text,
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

        cv2.imshow("Emotion Detection Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

# ==========================
# GUI WINDOW
# ==========================
root = Tk()

root.title("Emotion Detection System")

root.geometry("900x700")

root.configure(bg="white")

# Title
title = Label(
    root,
    text="Emotion Detection System",
    font=("Arial", 24, "bold"),
    bg="white",
    fg="darkblue"
)

title.pack(pady=20)

# Buttons Frame
button_frame = Frame(root, bg="white")

button_frame.pack(pady=20)

# Upload Button
upload_btn = Button(
    button_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    width=18,
    height=2
)

upload_btn.grid(row=0, column=0, padx=20)

# Webcam Button
webcam_btn = Button(
    button_frame,
    text="Open Webcam",
    command=open_webcam,
    font=("Arial", 14, "bold"),
    bg="blue",
    fg="white",
    width=18,
    height=2
)

webcam_btn.grid(row=0, column=1, padx=20)

# Image display
image_label = Label(root, bg="white")

image_label.pack(pady=20)

# Run GUI
root.mainloop()