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
        predictions = np.argmax(predictions, axis=-1)
        unique, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size

        prediction_percentages = {
            class_labels.get(int(k), "Unknown class"): (v / total_pixels * 100)
            for k, v in zip(unique, counts)
        }

        oil_spill_percentage = prediction_percentages.get("Oil Spill Pixels", 0)
        if oil_spill_percentage > 0:
            st.markdown("<span style='color: red;'>Oil Spill Detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: green;'>No Oil Spill Detected</span>", unsafe_allow_html=True)

        st.write("Percentage of each class in the image:", prediction_percentages)

        # Display the RGB image
        color_image = create_color_image(predictions)
        st.image(color_image, caption='Classified Image', use_column_width=True)

def create_color_image(predictions):
    class_to_color = {
        0: [0, 0, 0],      # Black for Sea Surface
        1: [0, 255, 255],  # Cyan for Oil Spill
        2: [255, 0, 0],    # Red for Look-alike
        3: [165, 42, 42],  # Brown for Ship
        4: [0, 128, 0]     # Green for Land
    }
    color_image = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    for class_value, color in class_to_color.items():
        color_image[predictions == class_value] = color
    return color_image


class_labels = {
    0: "Sea Surface Pixels",
    1: "Oil Spill Pixels",
    2: "Look-alike Pixels",
    3: "Ship Pixels",
    4: "Land Pixels",
}