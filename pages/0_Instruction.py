import streamlit as st

st.set_page_config(page_title="Instruction", page_icon="📈")
st.markdown("# Instruction")
st.write(
    """Follow the instruction to get weight estimation accurately."""
)

st.markdown('#### - Use 4x4 inch reference object.')
st.markdown('#### - Take picture with proper angle')

st.write(' ')
st.write('An example')
sample_image = 'sample_img/62_s_199_F.jpg'
st.image(sample_image, width = 300)
st.write('''Note: Reference object does not necessarily need to be
         a round shape object, you can use a square 4x4 inch paper. ''')
