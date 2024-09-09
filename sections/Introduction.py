import streamlit as st
from PIL import Image
import os
from config import project_dir

# Chemin absolu du répertoire du projet
project_dir = project_dir()

def intro():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:

        st.title("Machine Learning on Streamlit")
        st.write("Open the sidebar and choose a topic")
        image_path = os.path.join(project_dir, "images", "ML.jpg")
        st.image(Image.open(image_path))
