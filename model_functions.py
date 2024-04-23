import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

models = {
    'U-Net': load_model('Unet_model.h5'),
    'SegNet': load_model('SegNet_model.h5'),
    'FCN': load_model('FCN_model.h5')
}

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image, model_name):
    model = models[model_name]
    image = preprocess_image(image)
    predictions = model.predict(image)
    
    if predictions.ndim > 2:
        # Convert predictions to class indices per pixel
        predictions = np.argmax(predictions, axis=-1)
        unique, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size
        # Calculate percentage of each class
        prediction_percentages = {class_labels.get(int(k), "Unknown class"): (v / total_pixels * 100) for k, v in zip(unique, counts)}
        
        # Detect Oil Spill
        oil_spill_percentage = prediction_percentages.get("Oil Spill Pixels", 0)
        if oil_spill_percentage > 0:
            st.markdown("<span style='color: red;'>Oil Spill Detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: green;'>No Oil Spill Detected</span>", unsafe_allow_html=True)
        
        # Display percentages after the message
        st.write("Percentage of each class in the image:", prediction_percentages)

        # Find the most frequent class
        predicted_class = np.bincount(predictions.flatten()).argmax()
        predicted_class = int(predicted_class)  # Ensure it is a native Python int
    else:
        predicted_classes = np.argmax(predictions, axis=-1)
        predicted_class = int(predicted_classes[0])  # Convert to Python int

    return class_labels.get(predicted_class, "Unknown class")

class_labels = {
    0: "Sea Surface Pixels",
    1: "Oil Spill Pixels",
    2: "Look-alike Pixels",
    3: "Ship Pixels",
    4: "Land Pixels",
}
