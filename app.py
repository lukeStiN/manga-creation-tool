# streamlit run app.py 
import streamlit as st 
from utils import *

# st.set_page_config(layout='wide')

with st.sidebar :
    pattern = st.selectbox('Pattern', PATTERNS)

    col1, col2 = st.columns([5, 1])
    seaparator = col1.slider('Separator', 0, 80, 10, 1)
    color = col2.color_picker('Color')
    
    files = st.file_uploader(f'Image', FILES_TYPES, True)

    images = [stImage_2_arrayImage(f) for f in files]
    page = get_image_from_pattern(pattern, images).result(1, separator=seaparator, separator_color=hex_to_rgb(color))

    st.download_button(
        'Download', cv2.imencode('.png', page)[1].tobytes(), 
        file_name='page.png', mime='image/png',
        use_container_width=True, type='primary'
    )

st.image(cv2.resize(page, (512, 768)), channels='BGR')