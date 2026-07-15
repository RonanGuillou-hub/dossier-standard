"""
Tests pour src/data/make_dataset.py — partie météo.

Convention pytest : les fichiers doivent s'appeler test_*.py (ou *_test.py)
et les fonctions de test doivent commencer par test_ pour être découvertes
automatiquement — sinon `pytest`/`make test` les ignore silencieusement.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.make_dataset import fetch_weather_for_region


def test_fetch_weather_for_region_appelle_lapi_avec_les_bons_parametres():
    """
    Teste la construction de la requête SANS appeler la vraie API météo :
    un test unitaire ne doit pas dépendre du réseau (lent, flaky, et
    coûte un appel API à chaque exécution de la suite de tests).
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "daily": {
            "time": ["2026-06-01", "2026-06-02"],
            "temperature_2m_max": [22.0, 24.0],
            "temperature_2m_min": [14.0, 15.0],
            "precipitation_sum": [0.0, 2.5],
        }
    }
    fake_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=fake_response) as mock_get:
        result = fetch_weather_for_region("Sud", "2026-06-01", "2026-06-02")

    # Vérifie que l'API a été appelée avec les bonnes coordonnées (Marseille pour "Sud")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["latitude"] == 43.30
    assert kwargs["params"]["longitude"] == 5.37
    assert kwargs["params"]["start_date"] == "2026-06-01"
    assert kwargs["params"]["end_date"] == "2026-06-02"

    # Vérifie la structure du DataFrame retourné
    assert list(result.columns) == ["date", "region", "temperature_max", "temperature_min", "precipitation"]
    assert len(result) == 2
    assert result["region"].unique().tolist() == ["Sud"]


def test_fetch_weather_for_region_region_inconnue_leve_une_erreur():
    with pytest.raises(ValueError, match="Région inconnue"):
        fetch_weather_for_region("Atlantide", "2026-06-01", "2026-06-02")


@pytest.mark.integration
def test_fetch_weather_for_region_vrai_appel_api():
    """
    Test d'intégration : appelle la VRAIE API Open-Meteo. Volontairement
    séparé des tests unitaires ci-dessus via le marqueur @pytest.mark.integration
    — exclu par défaut de `make test` (voir pytest.ini), à lancer explicitement
    avec `pytest tests/ -m integration` quand tu veux vérifier la connectivité
    réelle à l'API.
    """
    result = fetch_weather_for_region("Sud", "2026-06-01", "2026-06-02")
    assert not result.empty
