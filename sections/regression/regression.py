import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso, TweedieRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import seaborn as sns



# Suppression de la fonction center()

def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def display_data_description(data):
    return data.describe()

def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def select_features(X, y):
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X = X.drop(columns=to_drop)

    corr_with_target = X.apply(lambda x: stats.pearsonr(x, y)[0])
    selected_features = corr_with_target[abs(corr_with_target) > 0.1].index.tolist()
    return X[selected_features]


def lazy_regressor(data):
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    X = data.drop('target', axis=1)
    y = data['target']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    diabetes_model = reg.fit(train_X, val_X, train_y, val_y)
    return diabetes_model

def plot_scatterplot(data, x_col, y_col):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.scatter(data[x_col], data[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    return fig

'''def plot_correlation_matrix(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Matrice de Corrélation')
    return fig'''

def train_and_evaluate_model(X, y, test_size, hyperparameter_tuning, selected_model=None, manual_hyperparameters=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    models = {
        'ElasticNet': (ElasticNet(), {'alpha': [0.0, 0.1, 1.0, 10, 0.5, 0.7, 0,4, 0.2], 'l1_ratio': [0.0, 0.1, 0.5, 0.7, 1.0]}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.1, 1.0, 10]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        'TweedieRegressor': (TweedieRegressor(), {'power': [0, 1, 1.5], 'alpha': [0.1, 0.5, 1.0]}),
        'LinearRegression': (LinearRegression(), {})
    }

    results = {}
    if hyperparameter_tuning == "Automatique":
        for name, (model, param_grid) in models.items():
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Perform cross-validation
            cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)

            results[name] = {
                'Train_Score': best_model.score(X_train, y_train),
                'Test_Score': best_model.score(X_test, y_test),
                'model': best_model,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std()
            }
    else:
        model = models[selected_model][0]
        model.set_params(**manual_hyperparameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        results[selected_model] = {
            'Train_Score': model.score(X_train, y_train),
            'Test_Score': model.score(X_test, y_test),
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }

    return results, X_test, y_test

def get_manual_hyperparameters(model_name):
    if model_name == "ElasticNet":
        alpha = st.number_input("Alpha", 0.01, 100.0, 1.0, 0.01)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
        return {"alpha": alpha, "l1_ratio": l1_ratio}
    elif model_name == "Ridge":
        alpha = st.number_input("Alpha", 0.01, 100.0, 1.0, 0.01)
        return {"alpha": alpha}
    elif model_name == "Lasso":
        alpha = st.number_input("Alpha", 0.01, 100.0, 1.0, 0.01)
        return {"alpha": alpha}
    elif model_name == "RandomForest":
        n_estimators = st.number_input("Nombre d'arbres", 10, 1000, 100, 10)
        max_depth = st.number_input("Profondeur maximale", 1, 100, 10, 1)
        return {"n_estimators": n_estimators, "max_depth": max_depth}
    elif model_name == "TweedieRegressor":
        power = st.slider("Power", 0.0, 2.0, 1.0, 0.1)
        alpha = st.number_input("Alpha", 0.01, 100.0, 1.0, 0.01)
        return {"power": power, "alpha": alpha}
    elif model_name == "LinearRegression":
        return {}


def plot_results(results):
    metrics = ['mse', 'r2', 'mae', 'rmse']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        values = [result[metric] for result in results.values()]
        axes[i].bar(results.keys(), values)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def plot_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Valeurs prédites')
    ax.set_title('Valeurs réelles vs Valeurs prédites')
    return fig


def regression_page():
    st.markdown("<h1 style='color: #003366;'>Prédiction de la progression du diabète</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    if uploaded_file:
        if 'data' not in st.session_state:
            st.session_state.data = load_data(uploaded_file)
        data = st.session_state.data

        st.subheader("Aperçu du jeu de données")
        st.dataframe(data.head())

        st.subheader("Description statistique du jeu de données")
        st.write(display_data_description(data))

        matrice_correlation = st.toggle("Matrice de Corrélation", value=False)
        if matrice_correlation:
            st.subheader("Matrice de Corrélation")
            st.dataframe(data.corr().style.background_gradient(cmap='coolwarm'))

            st.write("Sélection des variables explicatives:")
            st.write("1. Les variables fortement corrélées entre elles (> 0.95) sont considérées comme redondantes.")
            st.write("2. Les variables ayant une corrélation absolue avec la cible > 0.1 sont sélectionnées.")

        target_options = ["Choisir..."] + list(data.columns)
        target_column = st.selectbox("Sélectionnez la colonne cible pour la régression", target_options)

        if target_column and target_column != "Choisir...":
            show_scatterplot = st.toggle("Afficher le Scatterplot", value=False)
            if show_scatterplot:
                scatterplot_options = ["Choisir..."] + [col for col in data.columns if col != target_column]
                scatterplot_x = st.selectbox("Sélectionnez la variable pour l'axe X du scatterplot", scatterplot_options)
                if scatterplot_x and scatterplot_x != "Choisir...":
                    st.subheader(f"Scatterplot: {scatterplot_x} vs {target_column}")
                    fig = plot_scatterplot(data, scatterplot_x, target_column)
                    st.pyplot(fig)

            create_model = st.toggle("Créer le modèle", value=False)
            if create_model:
                test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20, 5) / 100
                hyperparameter_tuning = st.selectbox("Méthode de réglage des hyperparamètres", ["Automatique", "Manuel"])

                selected_model = None
                hyperparameters = None
                if hyperparameter_tuning == "Manuel":
                    selected_model = st.selectbox("Sélectionnez un modèle",
                        ["ElasticNet", "Ridge", "Lasso", "RandomForest", "TweedieRegressor", "LinearRegression"])
                    hyperparameters = get_manual_hyperparameters(selected_model)

                if st.button("Analyser et Entraîner le Modèle"):
                    with st.spinner("Traitement en cours..."):
                        X, y = preprocess_data(data, target_column)
                        X = select_features(X, y)
                        st.session_state.results, st.session_state.X_test, st.session_state.y_test = train_and_evaluate_model(
                            X, y, test_size, hyperparameter_tuning, selected_model, hyperparameters)
                    st.success("Modèle entraîné avec succès!")

                if 'results' in st.session_state:
                    st.subheader("Résultats de l'entraînement des modèles")
                    for name, result in st.session_state.results.items():
                        with st.expander(f"{name}"):
                            st.write(f"MSE: {result['mse']:.4f}")
                            st.write(f"R2 Score: {result['r2']:.4f}")
                            st.write(f"MAE: {result['mae']:.4f}")
                            st.write(f"RMSE: {result['rmse']:.4f}")
                            st.write(f"CV RMSE (mean ± std): {result['cv_rmse_mean']:.4f} ± {result['cv_rmse_std']:.4f}")

                            if hyperparameter_tuning == "Automatique":
                                st.subheader("Hyperparamètres choisis:")
                                for param, value in result['model'].get_params().items():
                                    st.write(f"{param}: {value}")
                            elif hyperparameter_tuning == "Manuel":
                                st.subheader("Hyperparamètres choisis:")
                                for param, value in hyperparameters.items():
                                    st.write(f"{param}: {value}")


                    best_model_name = min(st.session_state.results, key=lambda x: st.session_state.results[x]['mse'])
                    st.success(f"Meilleur modèle : {best_model_name}")

                    Comparaison_metriques = st.toggle("Comparaison des métriques", value=False)
                    if Comparaison_metriques:
                        st.subheader("Comparaison des métriques")
                        fig = plot_results(st.session_state.results)
                        st.pyplot(fig)

                    Comparaison_valeurs = st.toggle("Affichage des comparaisons", value=False)
                    if Comparaison_valeurs:
                        nb_rows_to_display = st.slider("Nombre de lignes à afficher", 1, 30, 5)
                        st.subheader(f"Comparaison entre les {nb_rows_to_display} premières lignes :")
                        best_model = st.session_state.results[best_model_name]['model']
                        comparison_df = pd.DataFrame({
                            "Valeurs réelles": st.session_state.y_test[:nb_rows_to_display],
                            "Valeurs prédites": best_model.predict(st.session_state.X_test)[:nb_rows_to_display]
                        })
                        st.dataframe(comparison_df)

                    plot_Predictions = st.toggle("Valeurs Prédites vs Réelles", value=False)
                    if plot_Predictions:
                        st.subheader("Valeurs Prédites vs Réelles")
                        best_model = st.session_state.results[best_model_name]['model']
                        fig_pred = plot_predictions(best_model, st.session_state.X_test, st.session_state.y_test)
                        st.pyplot(fig_pred)

                    if st.button("Sauvegarder le Meilleur Modèle"):
                        st.success("Modèle sauvegardé avec succès !")
        else:
            st.write("Veuillez sélectionner une colonne cible pour continuer l'analyse.")


if __name__ == "__main__":
    regression_page()
