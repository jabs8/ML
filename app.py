import streamlit as st

from sections.Introduction import intro
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Introduction", "Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    st.logo("images/logoDiabete.png", icon_image="images/logoDiabete.png")
    regression_page()
elif type_data == "Classification":
    st.logo("images/CouleursVins.jpg", icon_image="images/CouleursVins.jpg")
    classification_page()
elif type_data == "NailsDetection":
    st.logo("images/Ongles.jpg", icon_image="images/Ongles.jpg")
    nail_page()
elif type_data == "Introduction":
    st.logo("images/logoML.png", icon_image="images/logoML.png")
    intro()
