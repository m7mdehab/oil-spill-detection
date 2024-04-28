import streamlit as st

def main():
    st.title("Documentation on Oil Spill Detection Research")

    # Description section with the abstract
    st.header("Research Abstract")
    st.markdown("""
        This research paper delves into the application of computational methodologies
        for detecting oil spills in marine environments. Through the analysis of SAR data 
        and pattern recognition, the study employs machine learning and deep learning techniques 
        such as U-Net, SegNet, FCN, and DeeplabV3+ to enhance oil spill detection. 
        By reviewing extensive literature from fourteen papers, the paper assesses various approaches 
        and datasets to advance marine ecosystem conservation. The focus is on developing 
        more accurate and efficient methodologies to detect oil spills, 
        contributing significantly to environmental protection efforts.
    """)

    # Placeholder for the PDF link updated with the direct download link
    st.header("Full Research Paper")
    st.markdown("""
        Read the complete research paper for detailed insights into the methodologies, data analysis, and results. 
        [Download the full paper](https://drive.google.com/uc?export=download&id=1HzFIu3hNLMeeY__CL66ybPbM32SIHZ17)
    """)

if __name__ == "__main__":
    main()
