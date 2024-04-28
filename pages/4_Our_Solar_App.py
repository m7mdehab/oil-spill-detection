import streamlit as st

def main():
    st.title("Solar Cell Site Selection App")

    # Introduction to the Solar App
    st.header("About the Solar Site Selection App")
    st.markdown("""
        Our Solar Cell Site Selection mobile application assists users in evaluating 
        the potential success of specific locations for Solar PV power plant installation. 
        By inputting various location criteria, such as solar irradiation, slope, 
        proximity to transmission lines, and more, the app calculates the feasibility 
        and expected efficiency of solar energy generation at the given site.
    """)

    # Features of the Solar App
    st.subheader("Key Features")
    st.markdown("""
        - **Solar Irradiation Analysis**: Assess the solar energy potential based on historical and geographical data.
        - **Terrain Evaluation**: Consider the slope and terrain type which affect solar panel placement and construction.
        - **Infrastructure Assessment**: Evaluate proximity to essential infrastructure like transmission lines and roads.
        - **Recommendations**: Generate recommendations for the user on the criteria that need to improve to produce better results.
    """)

    # App download link
    st.header("Download the App")
    st.markdown("""
        Ready to explore solar potential in your area? Download our Solar Cell Site Selection app to get started:
        [Download the app](https://github.com/m7mdehab/oil-spill-detection/raw/main/Solar_App.apk)
        \n\n**Note**: This application is currently available only for Android devices.
    """)

if __name__ == "__main__":
    main()
