from sklearn.utils import shuffle

from .functions import *
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
import os
from config import project_dir

project_dir = project_dir()


def classification_page():
    Présentation, Visualisation, PreProcessing, LazyClassification, Modèles = st.tabs(
        ["Présentation", "Visualisation", "Pre-processing", "LazyClassification", "Modèles"])

    with Présentation:
        # Division de la page en 3 colonnes
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

            st.subheader("📐 Forme du dataset")
            checkbox = st.checkbox("Voir", key="shape")
            if checkbox:
                st.write(df.shape)

            st.subheader("📊 Description du dataset")
            checkbox = st.checkbox("Voir", key="description")
            if checkbox:
                describe(df)

            st.subheader("📃 Premières lignes du dataset")
            checkbox = st.checkbox("Voir", key="header")
            if checkbox:
                st.table(df.head())

            st.subheader("🎯 Analyse de la target")
            checkbox = st.checkbox("Voir", key="target")
            if checkbox:
                analyze_target(df)

            st.subheader("🔍 Matrice de corrélation")
            checkbox = st.checkbox("Voir", key="correlation")
            if checkbox:
                correlation_matrix(df)

    with PreProcessing:
        pass

    with Visualisation:
        pass

    with LazyClassification:
        # Lazy classificator
        lc = st.checkbox("Lazy Classificator")
        if lc:
            X = df.drop("target", axis=1)
            y = df["target"]
            #for rs in range(0,1,100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=69)
            '''clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            st.write(models)
            st.write(predictions)
'''
            for seed in range(5):  # Try with different seeds
                print(f"RandomState Seed: {seed}")

                # Shuffle data with a specific random state
                X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seed)

                # Initialize LazyClassifier with a specific random state
                clf = LazyClassifier(random_state=seed)

                # Train and evaluate the classifier
                models, predictions = clf.fit(X_train_shuffled, X_test, y_train_shuffled, y_test)

                # Print the performance of the models
                print(models)
                print("\n" + "=" * 40 + "\n")

    with Modèles:
        # Division de la page en 3 colonnes
        col1, col2 = st.columns([5,5])
        with col1:
            chosen_target = st.selectbox('Choose the Target Column', df.columns, index=len(df.columns) - 1)
            st.session_state['chosen_target'] = chosen_target
            X = df.drop(chosen_target, axis=1)
            y = df[chosen_target]
            target = st.slider('Test_Size', 0.1, 0.9, value=0.2)
            random_state = st.slider('Random_State', 0, 100, value=69)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=target, random_state=random_state)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)



            model = SVC()
            model.fit(X_train, y_train)
            # print prediction results
            y_pred = model.predict(X_test)

            st.title("1er modèle")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.write('Classification Report')
            st.table(report)

            # Matrice de confusion
            st.write(plot_confusion_matrix(y_test, y_pred))
            st.write(plot_class_distribution(y_pred, le))

            st.title("Recherche du meilleur modèle")
            # gridsearchcv

            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}

            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
            grid.fit(X_train, y_train)
            st.write("Meilleurs paramètres")
            st.write(grid.best_params_)
            st.write("Meilleurs estimateur")
            st.write(grid.best_estimator_)

            y_pred_best = grid.best_estimator_.predict(X_test)

            report_best = classification_report(y_test, y_pred_best, target_names=le.classes_, output_dict=True)
            st.write('New Classification Report')
            st.table(report_best)

            # Matrice de confusion
            st.write(plot_confusion_matrix(y_test, y_pred_best))
            st.write(plot_class_distribution(y_pred_best, le))

        with col2:
            # cross validation
            st.write("coucou")
            cvs = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring="accuracy")
            st.write(cvs)