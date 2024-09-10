import streamlit as st
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report


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

    st.write("Description et Analyse des colonnes")
    st.write("**alcohol**: degré d'alcool du vin")
    st.write("**malic_acid**: Acide malique, sa teneur influence l'acidité d'un vin")
    st.write("**ash**: cendres")
    st.write("**alcalinity_of_ash**: alcalinité des cendres")
    st.write("**magnesium**: magnésium en mg")
    st.write("**total_phenols**: Composition phénolique du vin. Ils contribuent à la couleur, à la qualité tannique, "
             "à la stabilité colloïdale et à l’aptitude de vieillissement des vins. ")
    st.write("**flavanoids**: molécules naturelles appartenant à la famille des polyphénols."
             "Contribuent à la couleur et à la sensation en bouche du vin")
    st.write("**nonflavanoid_phenols**: Autre phénols, contribuent au goût, à la couleur et à la sensation en bouche du vin")
    st.write("**proanthocyanins**: Autre molécules de la composition phénolique des vins.")
    st.write("**color_intensity**: Intensité de la couleur du vin")
    st.write("**hue**: teinte du vin")
    st.write("**od280/od315_of_diluted_wines**: Rapport  d'absorbance, qualifie la concentration de la protéine")
    st.write("**proline**: principal acide aminé du vin rouge et un élément important de la nutrition et de la saveur du vin.")
    st.write("**target**: 3 catégories de vin a prédire (sucré, amer et équilibré")



def analyze_target(df):
    categorical_columns = df.select_dtypes(['object', 'category']).columns
    if len(categorical_columns) > 0:
        target_col = st.selectbox("Choisissez la colonne cible (target)", df.columns, placeholder='target')
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

    df_vin_amer = df[df['target'] == 'Vin amer']
    df_vin_sucré = df[df['target'] == 'Vin sucré']
    df_vin_équilibré = df[df['target'] == 'Vin équilibré']
    st.subheader('Analyse de la target en fonction de certaines features')
    chosen_feature = st.selectbox('Choisir une feature : ', df.columns)
    fig, ax = plt.subplots()
    sns.distplot(df_vin_amer[chosen_feature], label="Vin amer")
    sns.distplot(df_vin_sucré[chosen_feature], label="Vin sucré")
    sns.distplot(df_vin_équilibré[chosen_feature], label="Vin équilibré")
    ax.legend()
    st.pyplot(fig)

def correlation_matrix(df):
    #fig, ax = plt.subplots()
    fig = sns.clustermap(df.drop("target", axis=1).corr())
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

def model_selected(chosen_model, X_train, X_test, y_train, y_test):
    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    if chosen_model == 'Support Vector Machines (SVC)':
        st.title("Support Vector Machines (SVC)")
        model = SVC()
        model.fit(X_train, y_train)
        # print prediction results
        y_pred = model.predict(X_test)
        st.header("Validation")
        cv_pick1 = st.selectbox("Choisir un mode de découpe", ["KFold", "StratifiedKFold", "ShuffleSplit"], key='ffzef')
        nb_découpe = st.selectbox("Nombre de découpe", range(3, 7, 1), key='11')
        if cv_pick1 == "KFold":
            cv = KFold(nb_découpe)
        elif cv_pick1 == "StratifiedKFold":
            cv = StratifiedKFold(nb_découpe)
        elif cv_pick1 == "ShuffleSplit":
            cv = ShuffleSplit(nb_découpe, test_size=0.2)

        cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        st.write("Validation score", cvs)

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.write('Classification Report')
        st.table(report)

        # Matrice de confusion
        st.write(plot_confusion_matrix(y_test, y_pred))
        st.write(plot_class_distribution(y_pred, le))
    elif chosen_model == 'Forêts aléatoires':
        st.title("Forêts aléatoires")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        # print prediction results
        y_pred = model.predict(X_test)
        st.header("Validation")
        cv_pick = st.selectbox("Choisir un mode de découpe", ["KFold", "StratifiedKFold", "ShuffleSplit"], key='20')
        nb_découpe = st.selectbox("Nombre de découpe", range(3, 7, 1), key='21')
        if cv_pick == "KFold":
            cv = KFold(nb_découpe)
        elif cv_pick == "StratifiedKFold":
            cv = StratifiedKFold(nb_découpe)
        elif cv_pick == "ShuffleSplit":
            cv = ShuffleSplit(nb_découpe, test_size=0.2)

        cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        st.write("Validation score", cvs)

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.write('Classification Report')
        st.table(report)

        # Matrice de confusion
        st.write(plot_confusion_matrix(y_test, y_pred))
        st.write(plot_class_distribution(y_pred, le))

    elif chosen_model == 'Gaussian Naive Bayes':
        st.title("Gaussian Naive Bayes")
        model = GaussianNB()
        model.fit(X_train, y_train)
        # print prediction results
        y_pred = model.predict(X_test)
        st.header("Validation")
        cv_pick = st.selectbox("Choisir un mode de découpe", ["KFold", "StratifiedKFold", "ShuffleSplit"], key='30')
        nb_découpe = st.selectbox("Nombre de découpe", range(3, 7, 1), key='31')
        if cv_pick == "KFold":
            cv = KFold(nb_découpe)
        elif cv_pick == "StratifiedKFold":
            cv = StratifiedKFold(nb_découpe)
        elif cv_pick == "ShuffleSplit":
            cv = ShuffleSplit(nb_découpe, test_size=0.2)

        cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        st.write("Validation score", cvs)

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.write('Classification Report')
        st.table(report)

        # Matrice de confusion
        st.write(plot_confusion_matrix(y_test, y_pred))
        st.write(plot_class_distribution(y_pred, le))