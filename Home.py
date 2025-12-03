
import PIL
import streamlit as st
from streamlit.logger import get_logger
from ultralytics import YOLO
import fun
import numpy as np

model_path = "models/ob.pt"
model_path2 = "models/sbest.pt"

def run():



    def internel(col1, col2, source_img, model_ob, model_sg):
    
        with col1:
            if source_img:
                # Opening the uploaded image
                uploaded_image = PIL.Image.open(source_img)
                # Adding the uploaded image to the page with a caption
                st.image(source_img,
                        caption="Uploaded Image",
                        use_container_width=True
                        )
                
                if st.button("Predict", type="primary"):
                    
                    try:
                        image_np = np.array(uploaded_image)

                        result_ob = model_ob.predict(image_np)
                        result_sg = model_sg.predict(image_np)

                        weight, img, checker = fun.mainloop(image_np, result_sg, result_ob)
                        transformed_img = PIL.Image.fromarray(img)

                        with col2:
                            st.image(transformed_img,
                                    caption='Detected Image',
                                    use_container_width=True
                                    )

                        if checker == 2 :
                            st.write(f'The weight is = {weight} kg')
                        
                        elif checker == 1:
                            st.write('⚠️ Warning: No reference object found for correct size measurement.')
                            st.write('⚠️ Warning: Without correct size measurement, measuring weight is invalid.')
                            st.write('⚠️ Consider using a 4 inch diameter reference object.')
                        
                        else:
                            st.write('Nothing found to measure.')



                    except Exception as ex:
                        st.error(
                            f"Nothing found to measure.")
                        st.error(ex)





    try:
        model_ob = YOLO(model_path)
        model_sg = YOLO(model_path2)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)


    st.set_page_config(
        page_title="Welcome to AI powered Cattle weight detection.",
        page_icon="🪄",
    )

    st.write("# Cattle Weight AI  🪄")


    st.markdown(
        """
        This app is specially built for predicting cow's weight from 2D image. \n
        Please refer to the Instructions page for proper weight determination.

        """
    )

    


    source_img = st.file_uploader(
        "Upload an image (jpg/jpeg only).", type=("jpg", "jpeg"))
    
    col1, col2 = st.columns(2)

    internel(col1, col2, source_img, model_ob, model_sg)

    st.sidebar.success("Select a demo image bellow to see how it's work.")

    with st.sidebar:
        sample_img1 = 'sample_img/52_s_117_F.jpg'
        sample_img2 = 'sample_img/62_s_199_F.jpg'
        sample_img3 = 'sample_img/63_s_129_F.jpg'
        sample_img4 = 'sample_img/56_s_209_F.jpg'

        st.image(sample_img1, use_container_width=True)
        if 'cl1' not in st.session_state:
            st.session_state.cl1 = False
        def cb1():
            st.session_state.cl1 = True
        st.button('Load Demo 1', on_click=cb1, use_container_width= True)
        if st.session_state.cl1:
            internel(col1, col2, sample_img1, model_ob, model_sg)


        st.image(sample_img2, use_container_width=True)
        if 'cl2' not in st.session_state:
            st.session_state.cl2 = False
        def cb2():
            st.session_state.cl2 = True
        st.button('Load Demo 2', key= 'f' , on_click=cb2, use_container_width= True)
        if st.session_state.cl2:
            internel(col1, col2, sample_img2, model_ob, model_sg)


        st.image(sample_img3, use_container_width=True)
        if 'cl3' not in st.session_state:
            st.session_state.cl3= False
        def cb3():
            st.session_state.cl3 = True
        st.button('Load Demo 3', on_click=cb3, use_container_width= True)
        if st.session_state.cl3:
            internel(col1, col2, sample_img3, model_ob, model_sg)

        
        st.image(sample_img4, use_container_width=True)
        if 'cl4' not in st.session_state:
            st.session_state.cl4 = False
        def cb4():
            st.session_state.cl4 = True
        st.button('Load Demo 4', on_click=cb4, use_container_width= True)
        if st.session_state.cl4:
            internel(col1, col2, sample_img4, model_ob, model_sg)

        

        


    
if __name__ == "__main__":
    run()
