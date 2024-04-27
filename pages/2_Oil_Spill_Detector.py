import streamlit as st
from PIL import Image
import model_functions
import time  # for simulating a delay in progress bar
import requests
from io import BytesIO

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        st.error(f"Failed to load image from URL: {url}. Error: {e}")
        return None

def main():
    st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊")
    st.write("# Oil Spill Detector 🌊🛢️")

    # Let the user upload their own image or select a sample image
    uploaded_file = st.file_uploader("Upload a SAR image, or select a sample image below:", type=["jpg", "jpeg", "png"])

    # Define sample images with URLs
    sample_images = {
        "Sample 1": "https://raw.githubusercontent.com/m7mdehab/oil-spill-detection/428fe74bab9f11e400dea26db50b33a637d80da4/Image%201.jpg",
        "Sample 2": "https://raw.githubusercontent.com/m7mdehab/oil-spill-detection/428fe74bab9f11e400dea26db50b33a637d80da4/Image%202.jpg",
        "Sample 3": "https://raw.githubusercontent.com/m7mdehab/oil-spill-detection/428fe74bab9f11e400dea26db50b33a637d80da4/Image%203.png",
        "Sample 4": "https://raw.githubusercontent.com/m7mdehab/oil-spill-detection/428fe74bab9f11e400dea26db50b33a637d80da4/Image%204.png",
        "Sample 5": "https://raw.githubusercontent.com/m7mdehab/oil-spill-detection/428fe74bab9f11e400dea26db50b33a637d80da4/Image%205.jpg"
    }

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        sample_selection = st.radio("Or choose from sample images:", list(sample_images.keys()))
        image_url = sample_images[sample_selection]
        image = load_image_from_url(image_url)
        if image:
            st.image(image, caption=f'Selected Image: {sample_selection}', use_column_width=True)

    # Ensure an image is selected or uploaded to enable model selection and detection
    if image:
        model_option = st.selectbox(
            'Which detection algorithm would you like to use?',
            ('DeeplabV3+ (Recommended)', 'U-Net', 'FCN', 'SegNet'),
            key='model_selector'
        )

        if st.button('Detect Oil Spill'):
            # Initialize progress bar
            progress_bar = st.progress(0)
            # Simulate the detection process and update progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.001)  # Simulate processing time
            # Call the prediction function
            label = model_functions.predict(image, model_option)
            progress_bar.progress(100)
            st.success('Detection complete!')

if __name__ == "__main__":
    main()