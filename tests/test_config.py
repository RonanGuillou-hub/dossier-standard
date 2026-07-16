"""Tests de src/config.py — chargement et cache de configs/config.yaml."""

from src.config import load_config


def test_load_config_retourne_un_dict_avec_les_sections_attendues():
    config = load_config()
    assert isinstance(config, dict)
    for section in ["paths", "data", "external", "model", "training", "mlflow", "huggingface", "s3", "api", "streamlit"]:
        assert section in config


def test_load_config_est_mis_en_cache():
    """@lru_cache doit retourner le même objet, pas relire le fichier à chaque appel."""
    assert load_config() is load_config()
