import streamlit as st

def main():
    st.title("Welcome to the Oil Spill Detection WebApp 🌊🛢️")
    st.markdown("""
        ## What does this webapp do?
        This Webapp uses deep learning models to detect oil spills in satellite images. 
        It allows users to upload Synthetic Aperture Radar (SAR) images, select from various detection models, 
        and visualize the detection results with annotated areas showing detected oil spills.
        
        ## How to use this app?
        - **Navigate to the Oil Spill Detection page**: This can be done via the sidebar where you'll find 'Oil Spill Detector'.
        - **Upload an image**: You can upload your SAR image on the 'Oil Spill Detector' page.
        - **Choose a detection model**: Select the model you wish to use for detecting oil spills.
        - **Analyze the image**: Click on the 'Detect Oil Spill' button to start the analysis.
        - **View results**: Examine the output which will display both the image with detected areas and statistical data regarding the detection.
        
    """)

if __name__ == "__main__":
    main()