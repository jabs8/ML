import streamlit as st
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def describe(df):
    st.table(df.describe())
    df_describe = df.describe()
    mean_col = df_describe.loc['mean']
    somme = sum(np.absolute(mean_col[i]) for i in range(len(mean_col))) / len(mean_col)
    if (df_describe.loc['mean'].all() == True) and (df_describe.loc['std'].all() == True) and (
            somme <= 0.001):
        st.write("✅ Données Standardisées")
    else:
        st.write("❌ Données non Standardisées")

def analyze_target(df):
    categorical_columns = df.select_dtypes(['object', 'category']).columns
    if len(categorical_columns) > 0:
        target_col = st.selectbox("Choisissez une colonne cible (target)", df.columns)
        st.write(df[target_col].value_counts())
        # Pie chart des valeurs cibles
        fig, ax = plt.subplots()
        df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")  # Pour éviter le texte de la colonne sur le graphique
        ax.set_title(f"Distribution en pourcentage de {target_col}")
        st.pyplot(fig)

    for col in categorical_columns:
        if col != target_col:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Répartition des valeurs dans {col}")
            ax.set_ylabel("Nombre d'occurrences")
            st.pyplot(fig)

def correlation_matrix(df):
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df.drop("target", axis=1).corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def pair_plot(df):
    st.write('Pairplot of Vine Dataset')
    pp = sns.pairplot(df, hue='target')
    # Adjust the size of the plot for better display in Streamlit
    plt.subplots_adjust(top=0.9)
    return st.pyplot(pp.figure)

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