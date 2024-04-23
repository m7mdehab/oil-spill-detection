import streamlit as st
import os

# Set environment variables to prevent TensorFlow from using GPU and TensorRT
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable all GPUs visible to TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs, only display errors

from PIL import Image
import model_functions

# This should be the very first Streamlit command used in your app, and it should not be in a function
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊")

def main():
    st.sidebar.write("## Explore Further")
    st.sidebar.markdown("[Documentation](https://docs.streamlit.io)")
    st.sidebar.markdown("[Community Forums](https://discuss.streamlit.io)")

    st.write("# Welcome to our Oil Spill Detector! 🌊🛢️")

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
            st.write(f'Prediction: {label}')

        st.markdown("""
        ### More Information
        - Want to learn more about how these models work? Check out [our documentation](https://docs.example.com).
        - Have questions or feedback? Join our [community forums](https://discuss.example.com).
        """)

if __name__ == "__main__":
    main()
