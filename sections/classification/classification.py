import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma.core import absolute
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os

# Chemin absolu du répertoire du projet
project_dir = "PycharmProjects/ML"

def classification_page():
    # Division de la page en 3 colonnes
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.title("Bienvenue dans la classification des vins")
        # Uploader le csv
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        image_path = os.path.join(project_dir, "images", "CouleursVins.jpg")
        st.image(Image.open(image_path))
        if uploaded_file:
            df = pd.read_csv(uploaded_file, index_col=0)

            # Dataset infos
            ds_info = st.checkbox("Voir l'analyse préliminaire du dataset")
            if ds_info:
                dataset_info(df)

            # Show pairplot
            pair_plot = st.checkbox("pairplot (take some time)")
            if pair_plot:
                pair_plot(df)

            chosen_target = st.selectbox('Choose the Target Column', df.columns, index=len(df.columns) - 1)
            st.session_state['chosen_target'] = chosen_target
            X = df.drop(chosen_target, axis=1)
            y = df[chosen_target]
            target = st.slider('Test_Size', 0.01, 0.99)
            random_state = st.slider('Random_State', 0, 100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=target, random_state=random_state)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)

            # Lazy classificator
            lc = st.checkbox("Lazy Classificator")
            if lc:
                lazyclass(X_train, X_test, y_train, y_test)

            model = SVC()
            model.fit(X_train, y_train)
            # print prediction results
            y_pred = model.predict(X_test)

            st.title("1er modèle")
            report = classification_report(y_test, y_pred, target_names=le.classes_)
            st.write('Classification Report')
            st.text(report)

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

            report_best = classification_report(y_test, y_pred_best, target_names=le.classes_)
            st.write('New Classification Report')
            st.text(report_best)

            # Matrice de confusion
            st.write(plot_confusion_matrix(y_test, y_pred_best))
            st.write(plot_class_distribution(y_pred_best, le))

            # cross validation


def pair_plot(df):
    st.write('Pairplot of Vine Dataset')
    pp = sns.pairplot(df, hue='target')
    # Adjust the size of the plot for better display in Streamlit
    plt.subplots_adjust(top=0.9)
    return st.pyplot(pp.figure)


def dataset_info(df):
    # 3. Afficher une description du dataset
    st.header("Description du Dataset")

    st.write(df.describe(include='all'))  # Inclure toutes les colonnes, y compris les non numériques

    ''' analyser données target et dire si c'est reg ou class ou autre'''

    df_describe = df.describe(include='all').drop(['target'], axis=1)
    mean_col = df_describe.loc['mean']
    somme = sum(absolute(mean_col[i]) for i in range(len(mean_col))) / len(mean_col)
    if (df_describe.loc['mean'].all() == True) and (df_describe.loc['std'].all() == True) and (somme <= 0.001):
        st.write("Données Standardisées")
    else:
        st.write("Données non Standardisées")
    # 4. Afficher les 5 premières lignes du dataset
    st.header("Aperçu des données")
    st.write(df.head())

    # 5. Sélectionner la colonne target (cible)
    categorical_columns = df.select_dtypes(['object', 'category']).columns
    if len(categorical_columns) > 0:
        target_col = st.selectbox("Choisissez une colonne cible (target)", categorical_columns)

        # Compte des valeurs dans la colonne cible
        st.subheader(f"Répartition des valeurs dans {target_col}")
        st.write(df[target_col].value_counts())

        # Pie chart des valeurs cibles
        fig, ax = plt.subplots()
        df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")  # Pour éviter le texte de la colonne sur le graphique
        ax.set_title(f"Distribution en pourcentage de {target_col}")
        st.pyplot(fig)

    # 6. Graphique pour les autres colonnes catégorielles
    st.header("Autres Visualisations")

    for col in categorical_columns:
        if col != target_col:
            st.subheader(f"Répartition des valeurs dans {col}")
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Répartition des valeurs dans {col}")
            ax.set_ylabel("Nombre d'occurrences")
            st.pyplot(fig)

    # 7. Graphique de corrélation pour les colonnes numériques (si elles existent)
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        df['numeric_targets'] = LabelEncoder().fit_transform(df['target'])
        st.subheader("Matrice de Corrélation des Colonnes Numériques")
        fig, ax = plt.subplots()
        sns.heatmap(df.drop("target", axis=1).corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)


def lazyclass(X_train, X_test, y_train, y_test):
    st.header('lazy')

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)


    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    st.write(models)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return fig


def plot_class_distribution(y_test, le):
    unique_classes, counts = np.unique(y_test, return_counts=True)
    fig = plt.figure(figsize=(10, 7))
    plt.bar(le.classes_, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.xticks(le.classes_)
    plt.show()
    return fig