import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_image_model(model_path):
    """ Load pre-trained CNN model for facial emotion recognition. """
    emotion_model = load_model(model_path)
    emotion_labels = ['angry', 'happy', 'sad', 'worried']
    return emotion_model, emotion_labels

def predict_emotion_from_image(image_path, emotion_model, emotion_labels):
    """ Predict emotion from an image using CNN. """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    prediction = emotion_model.predict(img)
    return emotion_labels[np.argmax(prediction)]
