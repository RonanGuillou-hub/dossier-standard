"""
Fixtures partagées par l'ensemble de la suite de tests.

trained_model / X_y sont en scope "session" : le pipeline n'est entraîné
qu'une seule fois pour toute la suite (coûteux à répéter à chaque test),
et n'est jamais muté par les tests qui l'utilisent (predict() ne modifie
pas le modèle).
"""

import pytest

from src.data.make_dataset import clean_data, generate_dirty_dataset
from src.models.train import build_model


@pytest.fixture
def raw_df():
    """Dataset synthétique brut, non nettoyé (voir generate_dirty_dataset)."""
    return generate_dirty_dataset()


@pytest.fixture
def clean_df(raw_df):
    """Dataset après nettoyage structurel (clean_data), sans données météo."""
    return clean_data(raw_df)


@pytest.fixture
def df_with_weather(clean_df):
    """clean_df enrichi de colonnes météo factices — évite tout appel réseau réel dans les tests."""
    df = clean_df.copy()
    df["temperature_max"] = 20.0
    df["temperature_min"] = 10.0
    df["precipitation"] = 0.0
    return df


@pytest.fixture(scope="session")
def X_y():
    """Version session-scope de df_with_weather, pour les fixtures coûteuses ci-dessous."""
    df = clean_data(generate_dirty_dataset())
    df["temperature_max"] = 20.0
    df["temperature_min"] = 10.0
    df["precipitation"] = 0.0
    X = df.drop(columns=["cible"])
    y = df["cible"].astype(int)
    return X, y


@pytest.fixture(scope="session")
def trained_model(X_y):
    """Pipeline entraîné une seule fois, réutilisé par tous les tests qui en ont besoin."""
    X, y = X_y
    model = build_model()
    model.fit(X, y)
    return model
