import streamlit as st
import os
from PIL import Image

from config import project_dir

# Chemin absolu du répertoire du projet
project_dir = project_dir()

def about_us_page():

    # Créer des colonnes pour centrer le contenu
    col1, col2, col3 = st.columns([1, 2, 1])

    # Utiliser la colonne du milieu pour centrer le contenu
    with col2:
        # Titre principal de la page
        st.title("Contact Us")
        # Afficher l'adresse e-mail
        st.subheader("Email :")
        st.write("dream.team@diginamic-formation.fr")

        # Afficher le lien vers le dépôt Git
        st.subheader("GitHub :")
        st.write("https://github.com/jabs8/ML")

        image_path = os.path.join(project_dir, "images", "teampicture.png")
        st.image(Image.open(image_path))
