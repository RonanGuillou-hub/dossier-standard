"""Tests de src/models/predict_model.py — chargement du modèle et prédiction."""

from unittest.mock import patch

import joblib
import pandas as pd
import pytest

from src.models.predict_model import enrich_with_weather, load_model, predict


@pytest.fixture(autouse=True)
def clear_weather_cache():
    """_weather_cache est un dict au niveau module, partagé entre tests -- reset systématique."""
    from src.models.predict_model import _weather_cache

    _weather_cache.clear()
    yield
    _weather_cache.clear()


@pytest.fixture
def fake_weather_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-07-16"]),
        "region": ["Nord"],
        "temperature_max": [22.5],
        "temperature_min": [14.0],
        "precipitation": [0.0],
    })


def test_load_model_local_leve_erreur_si_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(source="local", model_path=tmp_path)


def test_load_model_local_charge_correctement(tmp_path, trained_model):
    model_file = tmp_path / "model.joblib"
    joblib.dump(trained_model, model_file)

    loaded = load_model(source="local", model_path=tmp_path)
    assert hasattr(loaded, "predict")


def test_load_model_source_invalide_leve_value_error():
    with pytest.raises(ValueError, match="source inconnue"):
        load_model(source="invalide")


def test_load_model_mlflow_utilise_lalias_configure():
    """Vérifie que l'URI construite pour MLflow suit bien le format models:/<nom>@<alias>."""
    with patch("mlflow.set_tracking_uri") as mock_set_uri, \
         patch("mlflow.sklearn.load_model") as mock_load:
        mock_load.return_value = "modele_factice"
        result = load_model(source="mlflow")

    assert result == "modele_factice"
    mock_set_uri.assert_called_once()
    args, _ = mock_load.call_args
    assert args[0] == "models:/mon-projet-ml@champion"


def test_predict_ajoute_prediction_et_probabilite(trained_model, X_y):
    X, _ = X_y
    result = predict(trained_model, X.head(3))

    assert "prediction" in result.columns
    assert "probabilite" in result.columns
    assert len(result) == 3
    assert result["probabilite"].between(0, 1).all()


def test_predict_conserve_les_colonnes_dorigine(trained_model, X_y):
    X, _ = X_y
    result = predict(trained_model, X.head(3))
    for col in X.columns:
        assert col in result.columns


def test_enrich_with_weather_ajoute_les_colonnes_meteo(fake_weather_df):
    df = pd.DataFrame({
        "age": [35], "revenu": [42000], "anciennete_mois": [18],
        "categorie": ["A"], "region": ["Nord"],
    })
    with patch("src.models.predict_model.fetch_weather_for_region", return_value=fake_weather_df):
        result = enrich_with_weather(df)

    for col in ["date", "temperature_max", "temperature_min", "precipitation"]:
        assert col in result.columns
    assert result.loc[0, "temperature_max"] == 22.5


def test_enrich_with_weather_utilise_aujourdhui_si_date_absente(fake_weather_df):
    df = pd.DataFrame({
        "age": [35], "revenu": [42000], "anciennete_mois": [18],
        "categorie": ["A"], "region": ["Nord"],
    })
    with patch("src.models.predict_model.fetch_weather_for_region", return_value=fake_weather_df) as mock_fetch:
        enrich_with_weather(df)

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    args, _ = mock_fetch.call_args
    assert args[1] == today


def test_enrich_with_weather_un_seul_appel_par_region_date_distincte(fake_weather_df):
    df = pd.DataFrame({
        "age": [35, 40], "revenu": [42000, 50000], "anciennete_mois": [18, 24],
        "categorie": ["A", "B"], "region": ["Nord", "Nord"],  # même région, même date implicite
    })
    with patch("src.models.predict_model.fetch_weather_for_region", return_value=fake_weather_df) as mock_fetch:
        enrich_with_weather(df)

    assert mock_fetch.call_count == 1
