"""
Tests de l'API FastAPI (src/api/main.py).

Le modèle est injecté directement dans le cache (model_loader._cache)
plutôt que chargé depuis local/S3/MLflow — on teste ici le comportement
de l'API, pas le chargement du modèle (déjà couvert par
test_predict_model.py). fetch_weather_for_region est systématiquement
mocké : aucun test dans ce fichier n'appelle le réseau.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import model_loader


@pytest.fixture
def fake_weather_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-07-15"]),
        "region": ["Nord"],
        "temperature_max": [22.5],
        "temperature_min": [14.0],
        "precipitation": [0.0],
    })


@pytest.fixture(autouse=True)
def clear_weather_cache():
    """
    _weather_cache (src/api/main.py) est un dict au niveau module, donc
    partagé entre tous les tests — sans ce reset, un test pollue le
    suivant (cache déjà rempli, mock jamais appelé).
    """
    from src.api.main import _weather_cache

    _weather_cache.clear()
    yield
    _weather_cache.clear()


@pytest.fixture
def client(trained_model):
    """Injecte directement le modèle de test dans le cache, sans passer par local/s3/mlflow."""
    original_source = model_loader.CONFIG["api"]["model_source"]
    model_loader.CONFIG["api"]["model_source"] = "local"  # doit matcher _cache["source"] ci-dessous

    model_loader._cache["model"] = trained_model
    model_loader._cache["source"] = "local"
    model_loader._cache["loaded_at"] = "test"
    model_loader._cache["detail"] = None

    from src.api.main import app

    yield TestClient(app)

    # Reset pour ne pas polluer d'autres tests
    model_loader._cache["model"] = None
    model_loader._cache["source"] = None
    model_loader.CONFIG["api"]["model_source"] = original_source


VALID_PAYLOAD = {"age": 35, "revenu": 42000, "anciennete_mois": 18, "categorie": "A", "region": "Nord"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info_reflete_le_cache(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert r.json()["loaded"] is True
    assert r.json()["source"] == "local"


def test_predict_enrichit_avec_la_meteo_et_retourne_une_prediction(client, fake_weather_df):
    with patch("src.api.main.fetch_weather_for_region", return_value=fake_weather_df) as mock_fetch:
        r = client.post("/predict", json=VALID_PAYLOAD)

    assert r.status_code == 200
    assert "prediction" in r.json()
    assert "probabilite" in r.json()
    mock_fetch.assert_called_once()


def test_predict_utilise_aujourdhui_si_date_absente(client, fake_weather_df):
    with patch("src.api.main.fetch_weather_for_region", return_value=fake_weather_df) as mock_fetch:
        client.post("/predict", json=VALID_PAYLOAD)

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    args, _ = mock_fetch.call_args
    assert args[1] == today  # start_date == aujourd'hui


def test_predict_batch_reutilise_le_cache_meteo(client, fake_weather_df):
    """2 observations identiques (même région/date) -> un seul appel météo grâce au cache."""
    with patch("src.api.main.fetch_weather_for_region", return_value=fake_weather_df) as mock_fetch:
        r = client.post("/predict/batch", json={"observations": [VALID_PAYLOAD, VALID_PAYLOAD]})

    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2
    assert mock_fetch.call_count == 1


def test_predict_batch_vide_retourne_400(client):
    r = client.post("/predict/batch", json={"observations": []})
    assert r.status_code == 400


def test_predict_champ_manquant_retourne_422(client):
    r = client.post("/predict", json={"age": 35})
    assert r.status_code == 422


def test_predict_meteo_indisponible_retourne_502(client):
    with patch("src.api.main.fetch_weather_for_region", side_effect=RuntimeError("API météo down")):
        r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 502
