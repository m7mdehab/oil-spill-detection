import streamlit as st

def main():
    st.title("Model Comparison and Conclusions")

    st.markdown("""
        ## Model Performance Overview
        In the quest to detect oil spills efficiently, we have evaluated three key models: U-Net, FCN, and SegNet. Here's how they compare:

        ### U-Net
        U-Net has consistently shown excellent performance in tasks requiring precise segmentation. This model's ability to accurately map detailed features of oil spills and differentiate them from similar-looking phenomena is critical. The training outcomes for U-Net revealed high effectiveness in both training and testing, suggesting it excels in handling complex segmentation tasks where accuracy at the pixel level is crucial.

        ### FCN (Fully Convolutional Network)
        While versatile in handling different image sizes and efficient in processing, FCN showed slightly lesser accuracy than U-Net. It benefits from a depth of layers and a sequential connection that enables it to learn robust features; however, it might lack the fine-grained precision in localizing boundaries compared to U-Net. 

        ### SegNet
        Noted for its parameter efficiency due to its unique pooling indices used in up-sampling, SegNet performed comparably well. Its design helps in reducing the computational load, which is significant in deployment scenarios that require real-time processing or operate under hardware constraints.

        ## Comparative Insights
        - **Accuracy and Precision**: U-Net topped the charts with its detailed and precise segmentation capabilities, making it potentially the best choice for applications where boundary delineation is critical. SegNet, while slightly less precise than U-Net, still provided competitive accuracy.
        - **Efficiency and Speed**: SegNet offers an attractive balance between performance and efficiency, which could be decisive in scenarios where computational resources are limited or costs are a concern. FCN was also less demanding in terms of computational power than U-Net.
        - **Generalization to Test Data**: All models generalized well to test data, but U-Net showed slightly superior capability in adapting its learned features to unseen data.

        ## Model Accuracy Table
        | Model  | Training Accuracy | Validation Accuracy | Test Accuracy |
        |--------|-------------------|---------------------|---------------|
        | U-Net  | 95.07%            | 94.64%              | 95.07%        |
        | FCN    | 91.58%            | 91.91%              | 92.77%        |
        | SegNet | 90.61%            | 93.26%              | 92.74%        |
    """)

if __name__ == "__main__":
    main()
