import streamlit as st

st.set_page_config(page_title="Instruction", page_icon="📈")
st.markdown("# Instruction")
st.write(
    """Follow the instruction to get weight estimation accurately."""
)

st.markdown('#### - Use a 4 inch diameter reference object.')
st.markdown('#### - Take picture with proper angle.')
st.markdown('#### - Keep 1.5~2 meter distance from subject when taking the photo.')

st.write(' ')
st.write('An example')
sample_image = 'sample_img/62_s_199_F.jpg'
st.image(sample_image, width = 300)
st.write('''Note: Reference object does not necessarily need to be
         a round shape object, you can use a square 4x4 inch paper. ''')
