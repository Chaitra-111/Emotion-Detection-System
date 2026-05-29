# Emotion-Detection-System
# Emotion Detection System using CNN and OpenCV

## Overview

The Emotion Detection System is a real-time Artificial Intelligence and Computer Vision project developed using Python, TensorFlow, Keras, and OpenCV. The system is capable of detecting human facial emotions through both webcam input and uploaded images.

This project uses Haar Cascade Classifier for face detection and a Convolutional Neural Network (CNN) model for emotion classification. The model is trained on the FER2013 facial emotion dataset containing thousands of grayscale facial expression images.

The application supports:

* Real-time webcam-based emotion recognition
* Emotion detection from uploaded images
* GUI interface using Tkinter
* CNN-based deep learning prediction
* Face detection using OpenCV

---

# Features

* Real-time facial emotion detection using webcam
* Upload image and detect emotion
* Deep Learning-based CNN model
* Face detection using Haar Cascade Classifier
* User-friendly GUI interface
* Detection of 7 emotions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise

---

# Technologies Used

| Technology   | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Programming Language             |
| TensorFlow   | Deep Learning Framework          |
| Keras        | CNN Model Building               |
| OpenCV       | Computer Vision & Face Detection |
| NumPy        | Numerical Computation            |
| Tkinter      | GUI Development                  |
| Pillow       | Image Handling                   |
| Haar Cascade | Face Detection Algorithm         |
| CNN          | Emotion Classification           |

---

# Algorithms Used

## 1. Haar Cascade Classifier

The Haar Cascade algorithm is used to detect human faces from images and webcam frames. It identifies facial regions using Haar-like features and cascade classifiers.

### Purpose:

* Detect face coordinates
* Extract facial regions
* Pass detected faces to CNN model

---

## 2. Convolutional Neural Network (CNN)

The CNN model is used for emotion classification. The network learns facial features such as:

* Eye structure
* Smile patterns
* Eyebrow positions
* Facial muscle movements

### CNN Layers Used:

* Convolutional Layers
* ReLU Activation
* MaxPooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

---

## 3. Adam Optimizer

Adam optimizer is used to optimize CNN weights during training for faster convergence and better accuracy.

---

## 4. Categorical Crossentropy Loss Function

Used for multi-class emotion classification.

---

# Dataset Used

## FER2013 Dataset

The project is trained using the FER2013 dataset which contains:

* 48x48 grayscale facial images
* 7 emotion categories
* More than 35,000 facial images

### Emotion Classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

# Project Workflow

```text
Input Image / Webcam
        ↓
OpenCV Captures Frame
        ↓
Haar Cascade Detects Face
        ↓
Face Preprocessing
        ↓
48x48 Grayscale Conversion
        ↓
CNN Emotion Prediction
        ↓
Display Emotion Label
```

---

# Project Structure

```text
Emotion-Detection-System/
│
├── dataset/
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   └── emotion_model.h5
│
├── app.py
├── train.py
├── detect.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Chaitra-111/Emotion-Detection-System.git
```

## Move into Project Folder

```bash
cd Emotion-Detection-System
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\\Scripts\\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python train.py
```

This generates:

```text
models/emotion_model.h5
```

---

# Run the Application

```bash
python app.py
```

---

# Application Features

## Upload Image

* Upload facial image
* Detect emotion from image
* Display confidence score

## Open Webcam

* Real-time webcam detection
* Live emotion prediction
* Face tracking and classification

---

# Output Example

The application predicts emotions such as:

* Happy (0.92)
* Sad (0.71)
* Neutral (0.85)

---

# Future Improvements

* Improve model accuracy using Transfer Learning
* Add MobileNetV2 or ResNet architectures
* Deploy using Streamlit or Flask
* Add emotion analytics dashboard
* Add multiple face tracking
* Support video file detection
* Add cloud deployment

---

# Learning Outcomes

Through this project, the following concepts were implemented and understood:

* Deep Learning
* Computer Vision
* CNN Architecture
* Face Detection
* Image Preprocessing
* Real-Time Prediction
* GUI Development
* TensorFlow & OpenCV Integration

---

# Author

Developed by Chaitra Sri Polana

GitHub Repository:
https://github.com/Chaitra-111/Emotion-Detection-System
