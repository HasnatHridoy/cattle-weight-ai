
import PIL
import streamlit as st
from streamlit.logger import get_logger
import cv2
from ultralytics import YOLO
from functionality import image_pro, measurement, mainloop

model_path = "/home/vscode/.local/cattle-weight-ai/models/ob.pt"
model_path2 = "/home/vscode/.local/cattle-weight-ai/models/sbest.pt"

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

    with col1:
        if source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )
    
    try:
        model_ob = YOLO(model_path)
        model_sg = YOLO(model_path2)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)



    





if __name__ == "__main__":
    run()
