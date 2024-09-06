from inference_sdk import InferenceHTTPClient
import streamlit as st
from PIL import Image, ImageDraw


# Détails projet roboflow
ROBOFLOW_API_KEY = "xfMVPjHjsorrG6VOGPa7"
ROBOFLOW_PROJECT_ID = "project1-pfqn3"
ROBOFLOW_MODEL_VERSION = "2"

def nail_page():
    st.header("Bienvenue dans la détection d'ongle")
    # Titre de l'application
    st.title("Test de Modèle Roboflow")
    # Chargement de l'image par l'utilisateur
    uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Afficher l'image téléchargée
        image = Image.open(uploaded_image)
        st.image(image, caption="Image téléchargée")

        # Bouton pour faire une prédiction
        if st.button("Faire une prédiction"):
            # Appeler la fonction de prédiction
            result = predict_image(image)

            # Afficher les résultats de la prédiction
            if result:
                st.subheader("Résultats de la prédiction")
                #st.json(result)  # Afficher les résultats sous forme JSON

                time = result.get("time")
                # Ajouter du texte ou dessiner des boîtes sur l'image ici si nécessaire
                st.write(f"Prédiction(s) en {time} secondes")

                # Affichage des résultats de la prédiction:
                predictions = result.get("predictions", [])
                p = 0
                for pred in predictions:
                    confidence = pred["confidence"]
                    st.write(f"Prédiction : {p}, Confiance : {confidence}")
                    p += 1

                st.subheader("Résultats de la détection")

                # Extraire les prédictions
                detections = result.get("predictions")

                # Dessiner les prédictions sur l'image
                image_with_detections = draw_detections(image, detections)

                # Afficher l'image avec les annotations
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
        # Draw polygon with red outline
        draw.polygon(points, outline="red", width=8)
        # Ajouter une étiquette avec le nom de la classe et la confiance
        draw.text((x, y+100), f"Pred {p} ({confidence:.2f}%)",font_size=69, fill="black", align="down")
        p+=1
    return image