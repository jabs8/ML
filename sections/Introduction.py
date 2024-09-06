import streamlit as st
from PIL import Image

def intro():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:

        st.title("Machine Learning on Streamlit")
        st.write("Open the sidebar and choose a topic")
        st.image(Image.open("images/ML.jpg"))
