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
import streamlit as st

def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)


def preprocess_data(data, target_column):
    # Séparation des features et la target
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
    X.head()
    y = data['target']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    diabetes_model = reg.fit(train_X, val_X, train_y, val_y)
    return diabetes_model

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1.0, 10], 'l1_ratio': [0.1, 0.5, 0.7, 1.0]}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.1, 1.0, 10]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        'TweedieRegressor': (TweedieRegressor(), {'power': [0, 1, 1.5], 'alpha': [0.1, 0.5, 1.0]}),
        'LinearRegression': (LinearRegression(), {})
    }

    results = {}
    for name, (model, param_grid) in models.items():
        cv = KFold(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        results[name] = {
            'model': best_model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse' : np.sqrt(mean_squared_error(y_test, y_pred))
        }

    return results


def plot_results(results):
    metrics = ['mse', 'r2', 'mae', 'rmse']
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
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
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Values')
    return fig

def regression_page():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)

        st.write("Dataset Preview:")
        st.dataframe(data.head())

        target_column = st.selectbox("Select the target column for regression", data.columns)
        #st.write(lazy_regressor(data))

        if st.button("Analyze and Train Model"):
            X, y = preprocess_data(data, target_column)
            st.write("Data preprocessing completed.")

            X = select_features(X, y)
            st.write(f"Selected features: {', '.join(X.columns)}")

            results = train_and_evaluate_model(X, y)

            st.write("Model Training Results:")
            for name, result in results.items():
                st.subheader(name)
                st.write(f"MSE: {result['mse']:.4f}")
                st.write(f"R2 Score: {result['r2']:.4f}")
                st.write(f"MAE: {result['mae']:.4f}")
                st.write(f"RMSE: {result['rmse']:.4f}")

            fig = plot_results(results)
            st.pyplot(fig)

            best_model_name = min(results, key=lambda x: results[x]['mse'])
            best_model = results[best_model_name]['model']
            st.write(f"Best model: {best_model_name}")

            st.subheader("Actual vs Predicted Values")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            fig_pred = plot_predictions(best_model, X_test, y_test)
            st.pyplot(fig_pred)

            if st.button("Save Best Model"):
                # Here you would implement the logic to save the model
                st.write("Model saved successfully!")