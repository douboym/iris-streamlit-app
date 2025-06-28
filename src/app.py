import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# -------------------------------
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# -------------------------------

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "iris_classifier_model.joblib")
    features_path = os.path.join(base_path, "models", "feature_names.joblib")

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    return model, feature_names


@st.cache_data
def load_iris_data():
    return datasets.load_iris()

model, feature_names = load_model()
iris = load_iris_data()

# -------------------------------
# INTERFACE PRINCIPALE
# -------------------------------

st.title("🌸 Prédiction de l'espèce d'une fleur d'iris")
st.write("Ajuste les caractéristiques ci-dessous pour prédire l'espèce correspondante.")

# -------------------------------
# SAISIE UTILISATEUR
# -------------------------------

st.sidebar.header("Paramètres d'entrée")

def user_input_features():
    values = {
        feature_names[0]: st.sidebar.slider(feature_names[0], 4.3, 7.9, 5.4),
        feature_names[1]: st.sidebar.slider(feature_names[1], 2.0, 4.4, 3.4),
        feature_names[2]: st.sidebar.slider(feature_names[2], 1.0, 6.9, 1.3),
        feature_names[3]: st.sidebar.slider(feature_names[3], 0.1, 2.5, 0.2),
    }
    return pd.DataFrame([values])

df = user_input_features()

# -------------------------------
# AFFICHAGE DES DONNÉES SAISIES
# -------------------------------

st.subheader("🧾 Données saisies")
st.write(df)

# -------------------------------
# PRÉDICTION
# -------------------------------

prediction = model.predict(df)[0]
proba = model.predict_proba(df)[0]
predicted_class = iris.target_names[prediction]

st.subheader("🔮 Prédiction")
st.success(f"Espèce prédite : **{predicted_class.capitalize()}**")

st.subheader("📊 Probabilités")
st.bar_chart(pd.DataFrame([proba], columns=iris.target_names))

# -------------------------------
# VISUALISATION GRAPHIQUE
# -------------------------------

def create_scatter_plot():
    iris_df = pd.DataFrame(np.c_[iris.data, iris.target], columns=feature_names + ['target'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(ax=axes[0], data=iris_df, x=feature_names[0], y=feature_names[1],
                    hue="target", palette="viridis", alpha=0.6, legend=False)
    axes[0].scatter(df.iloc[0][feature_names[0]], df.iloc[0][feature_names[1]],
                    color='red', s=200, marker='*', label="Input")
    axes[0].set_title("Dimensions du sépale")
    axes[0].legend()

    sns.scatterplot(ax=axes[1], data=iris_df, x=feature_names[2], y=feature_names[3],
                    hue="target", palette="viridis", alpha=0.6, legend=False)
    axes[1].scatter(df.iloc[0][feature_names[2]], df.iloc[0][feature_names[3]],
                    color='red', s=200, marker='*', label="Input")
    axes[1].set_title("Dimensions du pétale")
    axes[1].legend()

    plt.tight_layout()
    return fig

st.subheader("📍 Visualisation")
st.pyplot(create_scatter_plot())

# -------------------------------
# IMPORTANCE DES VARIABLES
# -------------------------------

if hasattr(model, 'feature_importances_'):
    st.subheader("📈 Importance des caractéristiques")
    fi = pd.DataFrame({'Caractéristique': feature_names, 'Importance': model.feature_importances_})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fi.sort_values("Importance", ascending=False), x="Importance", y="Caractéristique", ax=ax)
    st.pyplot(fig)

# -------------------------------
# INFOS COMPLÉMENTAIRES
# -------------------------------

st.subheader("📚 À propos du dataset")
st.markdown("""
Le jeu de données **Iris** est un classique en machine learning, introduit par R.A. Fisher en 1936.

Il contient :
- 150 observations
- 3 espèces : *setosa*, *versicolor*, *virginica*
- 4 variables numériques mesurant les dimensions des sépales et pétales
""")
