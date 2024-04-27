import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

models = {
    'DeeplabV3+': load_tflite_model('deeplab_model.tflite'),
    'U-Net': load_model('Unet_model.h5'),
    'SegNet': load_model('SegNet_model.h5'),
    'FCN': load_model('FCN_model.h5')
}

def preprocess_image(image, target_size=(256, 256)):
    # Resize the image to the target size
    image = image.resize(target_size)
    
    # Convert the image to a numpy array and ensure type float32
    image = np.array(image, dtype=np.float32)
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Check if the image has fewer than 3 dimensions (e.g., grayscale)
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)  # Duplicate the grayscale data across three channels
    
    # Check for alpha channel in four-channel (RGBA) images
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # Drop the alpha channel
    
    # Ensure the image has a batch dimension
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    return image


def predict(image, model_name):
    model_name = model_name.replace(" (Recommended)", "")
    if model_name == 'DeeplabV3+':
        return predict_tflite(image, models[model_name])
    else:
        return predict_keras(image, model_name)


def predict_keras(image, model_name):
    model = models[model_name]
    image = preprocess_image(image)
    predictions = model.predict(image)
    return process_predictions(predictions)

def predict_tflite(image, interpreter):
    image = preprocess_image(image)  # Ensure image is correctly preprocessed
    
    print("Processed image shape:", image.shape)  # Print the shape of the image tensor to verify dimensions
    
    # Obtain input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to the input of the interpreter
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()  # Run inference

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return process_predictions(predictions)

def process_predictions(predictions):
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
