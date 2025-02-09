# Facial Emotion Recognition System

This project is a real-time facial emotion recognition system that uses a Convolutional Neural Network (CNN) to classify emotions from grayscale images of faces. It can detect and classify seven emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

---

## Table of Contents

1. [About the Project](#about-the-project)
2. [Key Features](#key-features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Real-Time Detection](#real-time-detection)
9. [Contributing](#contributing)
10. [License](#license)

---

## About the Project

This project is a **Real-Time Facial Emotion Recognition System** that uses a Convolutional Neural Network (CNN) to classify emotions from grayscale images of faces. The system can detect and classify seven emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**. It is designed to work in real-time using a webcam feed, making it ideal for applications in human-computer interaction, psychology research, and emotion-aware systems.

---

## Key Features

- **Deep Learning Model**: A CNN model trained on facial emotion data to classify emotions with high accuracy.
- **Real-Time Detection**: Detects faces and predicts emotions in real-time using a webcam.
- **Data Augmentation**: Enhances model performance by applying transformations like rotation, flipping, and zooming during training.
- **User-Friendly Interface**: Displays detected emotions with bounding boxes and labels on the video feed.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy

Install the required packages using:

```bash
pip install tensorflow opencv-python numpy
```

## Installation
##### 1. Clone the repository:
```
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition
```

#### 2.Download the Haar Cascade file for face detection:
```
wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

#### 3. Organize your dataset into the following directory structure:

```
data/
  train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Neutral/
    Sad/
    Surprise/
  test/
    Angry/
    Disgust/
    Fear/
    Happy/
    Neutral/
    Sad/
    Surprise/
```

## Usage

### Training the model
##### 1. Run the training script:
```
python train.py
```
##### The trained model weights will be saved as ```model_weights.h5```.

### Real-Time Emotion Detection
##### 1. Run the real-time detection script:
```
python test.py
```
##### 2. The webcam feed will open, and the system will detect faces and predict emotions in real-time.
##### 3. Press ```q``` to exit the application.


## Model Architecture
#### The CNN model consists of the following layers:
- **Convolutional Blocks**: Multiple convolutional layers with batch normalization, max-pooling, and dropout.
- **Fully Connected Layers**: A dense layer with 512 units and a softmax output layer for classification.

##### The model is trained using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy loss.

## Training
#### The CNN model consists of the following layers:
- ***Data Augmentation**: Rotation, flipping, shifting, shearing, and zooming.ected Layers**: A dense layer with 512 units and a softmax output layer for classification.
- **Callbacks**: Early stopping and learning rate reduction on plateau.
- **Epochs**: Up to 150 epochs, with early stopping based on validation accuracy.


## Real-Time Detection
#### The real-time detection script (```test.py```) performs the following steps:
- Captures video from the webcam.
- Detects faces using the Haar Cascade classifier.
- Preprocesses the detected face (resizing, normalization).
- Predicts the emotion using the trained model.
- Displays the predicted emotion on the video feed.


## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
 
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Thanks to the creators of TensorFlow, Keras, and OpenCV for providing the tools to build this project.

- The Haar Cascade classifier is provided by OpenCV.
