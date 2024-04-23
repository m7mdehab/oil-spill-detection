import streamlit as st
import os
from PIL import Image
import model_functions

# Set environment variables to prevent TensorFlow from using GPU and TensorRT
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable all GPUs visible to TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs, only display errors

# This should be the very first Streamlit command used in your app, and it should not be in a function
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊")

def main():
    st.write("# Oil Spill Detector 🌊🛢️")

    uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Selection box for choosing the model
        model_option = st.selectbox(
            'Which detection algorithm would you like to use?',
            ('U-Net', 'SegNet', 'FCN'),
            key='model_selector'
        )

        # Button to run the model
        if st.button('Detect Oil Spill'):
            label = model_functions.predict(image, model_option)

    # Add a message at the end of the page
    st.markdown("""**Want to learn more?**""")
    st.markdown("""Check out our [Our_Documentation](pages/3_Our_Documentation.py) for more in-depth information, detailed analysis of our results, and to learn more about our detection models.
    """)

if __name__ == "__main__":
    main()
