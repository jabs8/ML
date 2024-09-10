from inference_sdk import InferenceHTTPClient
import streamlit as st
from PIL import Image, ImageDraw
import os
from config import project_dir
import numpy as np
# Chemin absolu du répertoire du projet
project_dir = project_dir()

# Détails projet roboflow
ROBOFLOW_API_KEY = "xfMVPjHjsorrG6VOGPa7"
ROBOFLOW_PROJECT_ID = "project1-pfqn3"
ROBOFLOW_MODEL_VERSION = "2"

def nail_page():
    Présentation, Upload, Predictions = st.tabs(["Présentation", "Upload", "Predictions"])

    with Présentation:
        col1, col2 = st.columns([5,5])
        with col1:
            st.title("Bienvenue dans la détection d'ongles (Modèle Roboflow)")
            image_path = os.path.join(project_dir, "images", "Ongles.jpg")
            st.image(Image.open(image_path))

        with col2:
            st.header("Présentation")
            st.write("L'objectif est de détecter les ongles sur une image donnée.")
            st.write("Le modèle développé sur Roboflow est un modèle de deep learning ")
            st.write("Le dataset contient 17 images de base")

            st.header("Augmentations")
            st.write("Le dataset a été augmenté a 41 image grâce à l'outil d'augmentation de Roboflow comme suit:")
            st.write("Outputs per training example: 3")
            st.write("Rotation: Between -15° and +15°")
            st.write("Shear: ±10° Horizontal, ±10° Vertical")
            st.write("Brightness: Between -15% and +15%")
            st.write("Exposure: Between -10% and +10%")
            st.write("Blur: Up to 2.5px")

            st.header("Cliquez sur l'onglet Upload pour charger une image")

    with Upload:
         col1, col2 =st.columns([5,2])
         with col1:
            # Chargement de l'image par l'utilisateur
            uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                # Afficher l'image téléchargée
                image = Image.open(uploaded_image)
                st.image(image, caption="Image téléchargée")

                with col2:
                    st.subheader("Dimensions de l'image")
                    st.write('(hauteur, largeur, nombre de couleurs)')
                    st.write(np.array(image).shape)
                    st.subheader("Cliquez sur l'onglet Prédiction pour tester le modèle Roboflow")

    with Predictions:
        if st.button("Faire une prédiction"):
            result = predict_image(image)
            if result:
                col1, col2 = st.columns([2,5])
                with col1:
                    st.subheader("Résultats de la prédiction")
                    #st.json(result)
                    time = result.get("time")
                    st.write(f"Prédiction(s) en {time} secondes")
                    predictions = result.get("predictions", [])
                    p = 0
                    for pred in predictions:
                        confidence = pred["confidence"]
                        st.write(f"Prédiction : {p}, Confiance : {confidence}")
                        p += 1

                with col2:
                    st.subheader("Résultats sur l'image")
                    detections = result.get("predictions")
                    image_with_detections = draw_detections(image, detections)
                    st.image(image_with_detections, caption="Image avec détections", use_column_width=True)

# Fonction pour faire une prédiction via l'API Roboflow
def predict_image(image):
    # initialize the client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )

    return CLIENT.infer(image, model_id=f"{ROBOFLOW_PROJECT_ID}/{ROBOFLOW_MODEL_VERSION}")

# Fonction pour dessiner les résultats de détection sur l'image
def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    p=0
    for detection in detections:
        points = [(point["x"], point["y"]) for point in detection["points"]]
        x = detection["x"]
        y = detection["y"]
        confidence = detection["confidence"]
        # Adapt text size to image size
        xi, yi, ci = np.array(image).shape
        # Draw polygon with red outline
        draw.polygon(points, outline="red", width=round(1*xi/250))
        # Ajouter une étiquette avec le nom de la classe et la confiance
        draw.text((x, y), f"{p}", fill='black', align="down", font_size=10*xi/250)
        p+=1
    return image