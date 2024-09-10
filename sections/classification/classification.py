from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from .functions import *
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.svm import SVC
import os
from config import project_dir

project_dir = project_dir()


def classification_page():
    Présentation, Visualisation, PreProcessing, Modèles = st.tabs(
        ["Présentation", "Visualisation", "Pre-processing", "Modèles"])

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

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=target, random_state=random_state)

        # Standardisation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        # Model
        #cv_pick = st.selectbox("Choisir un mode de découpe", ["KFold", "StratifiedKFold", "ShuffleSplit"])
        model = SVC()
        model.fit(X_train, y_train)
        # print prediction results
        y_pred = model.predict(X_test)


        # Matrice de confusion

        st.header("Validation")
        cv_pick = st.selectbox("Choisir un mode de découpe", ["KFold", "StratifiedKFold", "ShuffleSplit"])
        nb_découpe = st.selectbox("Nombre de découpe", range(3,7,1))
        if cv_pick == "KFold":
            cv = KFold(nb_découpe)
        elif cv_pick == "StratifiedKFold":
            cv = StratifiedKFold(nb_découpe)
        elif cv_pick == "ShuffleSplit":
            cv = ShuffleSplit(nb_découpe, test_size=0.2)

        cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        st.write("Validation score", cvs)
        col1, col2 = st.columns([5, 5])
        with col1:
            st.title("1er modèle")
            st.write({
                "C": 1,
                "gamma": 'scale',
                "kernel": "rbf"
            })
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.write('Classification Report')
            st.table(report)

            # Matrice de confusion
            st.write(plot_confusion_matrix(y_test, y_pred))
            st.write(plot_class_distribution(y_pred, le))

        with col2:
            st.title("Recherche du meilleur modèle")
            # gridsearchcv

            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf', 'poly']}

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


