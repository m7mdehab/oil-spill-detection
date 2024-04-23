import streamlit as st

def main():
    st.title("How It Works")
    
    st.markdown("""
        ## Deep Learning Models for Detection
        This application uses state-of-the-art deep learning models to identify and classify various features in satellite images. The models used include:
        
        - **U-Net**: Primarily used for precise segmentation tasks. It's effective in distinguishing complex features in images.
        - **SegNet**: Known for its efficiency in segmenting image pixels into categorically distinct classes.
        - **FCN (Fully Convolutional Network)**: Adapts classical neural networks for pixel-wise segmentation.
        
        ## Classes and RGB Color Coding
        The models classify the image pixels into five different classes each represented with a specific color:
        
        - **Sea Surface Pixels** (Black): Representing water surfaces without any contamination.
        - **Oil Spill Pixels** (Cyan): Indicating the presence of oil spills.
        - **Look-alike Pixels** (Red): Pixels that may resemble oil spills but are not.
        - **Ship Pixels** (Brown): Pixels that represent ships.
        - **Land Pixels** (Green): Pixels representing land surfaces.
        
        ## Outputs
        When an image is processed, the model predicts a class for each pixel. The output is a color-coded image where each pixel's color corresponds to its classified category. This allows for quick visual assessment of areas affected by oil spills and other features.
        
        The percentage of each class present in the image is also displayed, providing a quantitative measure of the analysis.
                
        Select the 'Oil Spill Detector' from the sidebar to get started and classify your own SAR images!

    """)

if __name__ == "__main__":
    main()
