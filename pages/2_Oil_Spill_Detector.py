import streamlit as st
from PIL import Image
import model_functions
import time  # for simulating a delay in progress bar

def main():
    st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊")
    st.write("# Oil Spill Detector 🌊🛢️")

    # Let the user upload their own image or select a sample image
    uploaded_file = st.file_uploader("Upload a SAR image, or select a sample image below:", type=["jpg", "jpeg", "png"])

    # Define sample images (assuming these are paths to images in your project directory)
    sample_images = {
        "Sample 1": "/workspaces/oil-spill-detection/Image 1.jpg",
        "Sample 2": "/workspaces/oil-spill-detection/Image 2.jpg",
        "Sample 3": "/workspaces/oil-spill-detection/Image 3.png",
        "Sample 4": "/workspaces/oil-spill-detection/Image 4.png",
        "Sample 5": "/workspaces/oil-spill-detection/Image 5.jpg",
    }

    # User can choose from sample images if no file is uploaded
    if not uploaded_file:
        sample_selection = st.radio("Or choose from sample images:", list(sample_images.keys()))
        image_path = sample_images[sample_selection]
        image = Image.open(image_path)
        st.image(image, caption=f'Selected Image: {sample_selection}', use_column_width=True)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Ensure an image is selected or uploaded to enable model selection and detection
    if image:
        model_option = st.selectbox(
            'Which detection algorithm would you like to use?',
            ('DeeplabV3+ (Recommended)', 'U-Net', 'SegNet', 'FCN'),
            key='model_selector'
        )

        if st.button('Detect Oil Spill'):
            # Initialize progress bar
            progress_bar = st.progress(0)
            # Simulate the detection process and update progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.005)  # Simulate processing time
            # Call the prediction function
            st.success('Detection complete!')
            label = model_functions.predict(image, model_option)
            progress_bar.progress(100)

if __name__ == "__main__":
    main()