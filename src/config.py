"""
Chargement centralisé de la configuration du projet (configs/config.yaml).

Ce module ne contient QUE des paramètres non-secrets : hyperparamètres,
chemins, noms de ressources. Les secrets (tokens, credentials) restent
en variables d'environnement, jamais dans ce fichier — voir os.environ
dans train.py / predict_model.py pour HF_TOKEN, MLFLOW_TRACKING_URI, etc.
"""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


@lru_cache
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Charge et met en cache le fichier de configuration YAML."""
    with open(path) as f:
        return yaml.safe_load(f)
