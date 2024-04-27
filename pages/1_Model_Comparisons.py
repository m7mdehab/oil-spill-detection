import streamlit as st

def main():
    st.title("Model Comparison and Conclusions")

    st.markdown("""
        ## Model Performance Overview
        In the quest to detect oil spills efficiently, we have evaluated four key models: DeeplabV3+, U-Net, FCN, and SegNet. Here's how they compare:

        ### DeeplabV3+
        DeeplabV3+ has excelled in segmentation accuracy and efficiency, standing out for its advanced deep learning capabilities that enhance semantic image segmentation. It has shown exceptional results in adapting to complex and diverse datasets, providing highly accurate and precise segmentations, which makes it the recommended choice for high-stakes applications.

        ### U-Net
        U-Net has consistently shown excellent performance in tasks requiring precise segmentation. This model's ability to accurately map detailed features of oil spills and differentiate them from similar-looking phenomena is critical. The training outcomes for U-Net revealed high effectiveness in both training and testing, suggesting it excels in handling complex segmentation tasks where accuracy at the pixel level is crucial.

        ### FCN (Fully Convolutional Network)
        While versatile in handling different image sizes and efficient in processing, FCN showed slightly lesser accuracy than U-Net. It benefits from a depth of layers and a sequential connection that enables it to learn robust features; however, it might lack the fine-grained precision in localizing boundaries compared to U-Net.

        ### SegNet
        Noted for its parameter efficiency due to its unique pooling indices used in up-sampling, SegNet performed comparably well. Its design helps in reducing the computational load, which is significant in deployment scenarios that require real-time processing or operate under hardware constraints.

        ## Comparative Insights
        - **Accuracy and Precision**: DeeplabV3+ and U-Net topped the charts with detailed and precise segmentation capabilities, making them potentially the best choices for applications where boundary delineation is critical. While DeeplabV3+ provides the highest accuracy, U-Net follows closely with robust performance.
        - **Efficiency and Speed**: SegNet offers an attractive balance between performance and efficiency, which could be decisive in scenarios where computational resources are limited or costs are a concern. FCN and DeeplabV3+ also showcase significant efficiency, handling complex tasks with lower computational demand compared to traditional models.
        - **Generalization to Test Data**: All models generalized well to test data, but DeeplabV3+ showed the superior capability in adapting its learned features to unseen data, closely followed by U-Net.

                
        | Model         | Test Loss | Test Accuracy | Precision | Recall | IOU     |
        |---------------|-----------|---------------|-----------|--------|---------|
        | **DeeplabV3+**| 0.115     | 96.25%        | 96.25%    | 96.25% | 92.77%  |
        | **U-Net**     | 0.215     | 93.55%        | 94.07%    | 92.99% | 40.00%  |
        | **FCN**       | 0.174     | 93.23%        | 93.23%    | 93.23% | 87.32%  |
        | **SegNet**    | 0.223     | 92.87%        | 92.87%    | 92.87% | 86.70%  |


    """)

if __name__ == "__main__":
    main()
