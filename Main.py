import streamlit as st
from streamlit.logger import get_logger

def run():
    st.set_page_config(
        page_title="Welcome to AI powered Cattle weight detection.",
        page_icon="🪄",
    )

    st.write("# Cattle Weight AI  🪄")

    st.sidebar.success("No reference image? Select a demo above.")

    st.markdown(
        """
        This app is specially built for predicting cow's weight from 2D image.
        Please upload a picture with reference object to procced.

        """
    )

    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    
    col1, col2 = st.columns(2)

    





if __name__ == "__main__":
    run()
