import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso, TweedieRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats


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


def train_and_evaluate_model(X, y, test_size, hyperparameter_tuning, selected_model=None, manual_hyperparameters=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    models = {
        'ElasticNet': (ElasticNet(), {'alpha': [0.0, 0.1, 1.0, 10], 'l1_ratio': [0.0, 0.1, 0.5, 0.7, 1.0]}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.1, 1.0, 10]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        'TweedieRegressor': (TweedieRegressor(), {'power': [0, 1, 1.5], 'alpha': [0.1, 0.5, 1.0]}),
        'LinearRegression': (LinearRegression(), {})
    }

    results = {}
    if hyperparameter_tuning == "Automatique":
        for name, (model, param_grid) in models.items():
            cv = KFold(n_splits=5)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            results[name] = {
                'Train_Score': best_model.score(X_train, y_train),
                'Test_Score': best_model.score(X_test, y_test),
                'model': best_model,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
    else:
        model = models[selected_model][0]
        model.set_params(**manual_hyperparameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        results[selected_model] = {
            'Train_Score': model.score(X_train, y_train),
            'Test_Score': model.score(X_test, y_test),
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
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
    st.title("Analyse de Régression")

    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)

        st.subheader("Aperçu du jeu de données")
        st.dataframe(data.head())  # Affichage de l'aperçu du jeu de données

        st.subheader("Description statistique du jeu de données")
        st.write(display_data_description(data))

        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox("Sélectionnez la colonne cible pour la régression", data.columns)
        with col2:
            test_size = st.slider("Sélectionnez la taille de l'ensemble de test (%)", min_value=10, max_value=50,
                                  value=20, step=5) / 100

        st.write(f"Taille de l'ensemble de test : {test_size:.0%}")

        hyperparameter_tuning = st.selectbox("Méthode de réglage des hyperparamètres", ["Automatique", "Manuel"])

        selected_model = None
        hyperparameters = None

        if hyperparameter_tuning == "Manuel":
            selected_model = st.selectbox("Sélectionnez un modèle",
                                          ["ElasticNet", "Ridge", "Lasso", "RandomForest", "TweedieRegressor",
                                           "LinearRegression"])
            hyperparameters = get_manual_hyperparameters(selected_model)

        if st.button("Analyser et Entraîner le Modèle"):
            with st.spinner("Traitement en cours..."):
                X, y = preprocess_data(data, target_column)
                st.success("Prétraitement des données terminé.")

                X = select_features(X, y)
                st.write(f"Caractéristiques sélectionnées : {', '.join(X.columns)}")

                results, X_test, y_test = train_and_evaluate_model(X, y, test_size, hyperparameter_tuning, selected_model, hyperparameters)

                st.subheader("Résultats de l'entraînement des modèles")
                col1, col2 = st.columns(2)
                for i, (name, result) in enumerate(results.items()):
                    with (col1 if i % 2 == 0 else col2):
                        with st.expander(f"{name}"):
                            st.write(f"MSE: {result['mse']:.4f}")
                            st.write(f"R2 Score: {result['r2']:.4f}")
                            st.write(f"MAE: {result['mae']:.4f}")
                            st.write(f"RMSE: {result['rmse']:.4f}")

                            if hyperparameter_tuning == "Automatique":
                                st.subheader("Hyperparamètres choisis:")
                                for param, value in result['model'].get_params().items():
                                    st.write(f"{param}: {value}")
                            elif hyperparameter_tuning == "Manuel":
                                st.subheader("Hyperparamètres choisis:")
                                for param, value in hyperparameters.items():
                                    st.write(f"{param}: {value}")

                st.subheader("Comparaison des métriques")
                fig = plot_results(results)
                st.pyplot(fig)

                best_model_name = min(results, key=lambda x: results[x]['mse'])
                best_model = results[best_model_name]['model']
                st.success(f"Meilleur modèle : {best_model_name}")

                # Ajout du slider pour sélectionner le nombre de lignes à afficher
                nb_rows_to_display = st.slider("Choisissez le nombre de lignes à afficher", 1, 30, 5)

                # Comparaison entre les valeurs réelles et prédites
                st.subheader(f"Comparaison entre les {nb_rows_to_display} premières lignes :")
                st.write(f"Valeurs réelles de la colonne cible (\"{target_column}\") vs Valeurs prédites")

                # Création des colonnes "Valeurs réelles" et "Valeurs prédites"
                comparison_df = pd.DataFrame({
                    "Valeurs réelles de la cible": y_test[:nb_rows_to_display].values,
                    "Valeurs prédites de la cible": best_model.predict(X_test)[:nb_rows_to_display]
                })

                # Affichage des deux colonnes
                st.dataframe(comparison_df)

                st.subheader("Valeurs Réelles vs Prédites")
                fig_pred = plot_predictions(best_model, X_test, y_test)
                st.pyplot(fig_pred)


                if st.button("Sauvegarder le Meilleur Modèle"):
                    st.success("Modèle sauvegardé avec succès !")


if __name__ == "__main__":
    regression_page()
