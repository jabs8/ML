from imblearn.over_sampling import SMOTE
from .functions import *
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report

from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from config import project_dir

project_dir = project_dir()


def classification_page():
    Présentation, Visualisation, Modèles = st.tabs(
        ["Présentation", "Visualisation", "Modèles"])

    with Présentation:
        col1, col2, col3 = st.columns([2,5,2])
        with col2:
            st.title("Bienvenue dans la classification des vins")

            # Uploader le csv
            df = pd.read_csv(f"{project_dir}data/vin.csv", index_col=0)
            '''uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)'''

            image_path = os.path.join("", "images", "CouleursVins.jpg")
            st.image(Image.open(image_path))

            st.header("📐 Forme du dataset")
            checkbox = st.checkbox("Voir", key="shape")
            if checkbox:
                x, y = df.shape
                st.write(f"Il y a :green[{x}] lignes et :green[{y}] colonnes")

            st.header("📊 Description du dataset")
            checkbox = st.checkbox("Voir", key="description")
            if checkbox:
                describe(df)

            st.header("📃 Premières lignes du dataset")
            checkbox = st.checkbox("Voir", key="header")
            if checkbox:
                st.table(df.head())

            st.header("❓ Type des colonnes")
            checkbox = st.checkbox("Voir", key="type")
            if checkbox:
                st.write(df.dtypes.value_counts())
                st.table(df.dtypes)

            st.header("♒ Check des doublons")
            checkbox = st.checkbox("Voir", key="doublons")
            if checkbox:
                st.write(df.duplicated().sum())

            st.header("🚫 Check des Valeurs manquantes")
            checkbox = st.checkbox("Voir", key="manquantes")
            if checkbox:
                df2= df.isnull().sum()
                if df2.sum()==0:
                    st.write("Pas de valeurs manquantes ✅")
                else : st.write(f"Il y a {df2.sum()} valeurs manquantes ⏬")
                st.table(df.isnull().sum())

    with Visualisation:
        col1, col2, col3 = st.columns([2, 5, 2])
        with col2:
            st.header("🎯 Analyse de la target")
            checkbox = st.checkbox("Voir", key="target")
            if checkbox:
                analyze_target(df)


            st.header("🎰 Analyse des features")
            checkbox = st.checkbox("Voir", key="features")
            if checkbox:
                for col in df.select_dtypes('float64'):
                    fig, ax = plt.subplots()
                    sns.distplot(df[col])
                    st.pyplot(fig)

            st.header("🔍 Corrélations des features")
            checkbox = st.checkbox("Voir", key="correlation")
            if checkbox:
                st.write('Cluster map')
                correlation_matrix(df)
                df_drop = df.drop("target", axis=1)
                for col in df_drop:
                    st.write(df_drop.corr()[col].sort_values(ascending=False))

    with Modèles:

        chosen_target = st.selectbox('Choose the Target Column', df.columns, index=len(df.columns) - 1, key="prepro")
        st.session_state['chosen_target'] = chosen_target
        X = df.drop(chosen_target, axis=1)
        y = df[chosen_target]
        target = st.slider('Test_Size', 0.1, 0.9, value=0.2)
        random_state = st.slider('Random_State', 0, 100, value=69)

        # Rééquilibrage target
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        st.write(y_res.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=target, random_state=random_state)

        col1, col2, col3 = st.columns([5,1,5])
        with col1:
            chosen_model1 = st.selectbox('Choisir un modèle', ['Support Vector Machines (SVC)', 'Forêts aléatoires',
                                                               'Gaussian Naive Bayes'], key='1dd')
            st.session_state['chosen_target'] = chosen_model1

            model_selected(chosen_model1, X_train, X_test, y_train, y_test)


        with col3:
            chosen_model2 = st.selectbox('Choisir un modèle', ['Support Vector Machines (SVC)', 'Forêts aléatoires',
                                                               'Gaussian Naive Bayes'], key='2gt')
            st.session_state['chosen_target'] = chosen_model2

            model_selected(chosen_model2, X_train, X_test, y_train, y_test)


