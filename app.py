from os import getcwd

import streamlit as st

from sections.Introduction import intro
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page
from sections.about_us.about_us import about_us_page
import os
from config import project_dir

project_dir = project_dir()


st.set_page_config(
    page_title="Playground ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Introduction", "Regression", "Classification", "NailsDetection", "About Us"]
)



if type_data == "Regression":
    image_path = os.path.join(project_dir, "images", "logoDiabete.png")
    st.logo(image_path, icon_image=image_path)
    regression_page()
elif type_data == "Classification":
    image_path = os.path.join(project_dir, "images", "CouleursVins.jpg")
    st.logo(image_path, icon_image=image_path)
    classification_page()
elif type_data == "NailsDetection":
    image_path = os.path.join(project_dir, "images", "Ongles.jpg")
    st.logo(image_path, icon_image=image_path)
    nail_page()
elif type_data == "Introduction":
    image_path = os.path.join(project_dir, "images", "logoML.png")
    st.logo(image_path, icon_image=image_path)
    intro()
elif type_data == "About Us":
    image_path = os.path.join(project_dir, "images", "teampicture.png")
    st.logo(image_path, icon_image=image_path)
    about_us_page()

