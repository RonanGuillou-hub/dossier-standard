"""
Génération de prédictions sur des données fraîches, à partir du modèle
déjà entraîné (aucun entraînement n'a lieu ici).

Conçu pour tourner directement sur un runner GitHub Actions (CPU) :
contrairement à l'entraînement, l'inférence avec ce modèle ne nécessite
pas de GPU, donc pas besoin de passer par HuggingFace Jobs ici.

Flux :
    1. Téléchargement du modèle entraîné depuis le Hub HuggingFace (ou local)
    2. Chargement des données fraîches
    3. Prédiction (le pipeline sklearn complet - FeatureEngineer +
       ColumnTransformer inclus - gère tout automatiquement)
    4. Sauvegarde des résultats
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
DEFAULT_OUTPUT = Path("reports/predictions.csv")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "mon-user/mon-modele")
HF_TOKEN = os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU MODÈLE (déjà entraîné, aucun apprentissage ici)
# ---------------------------------------------------------------------------
def load_model(source: str = "hub", model_path: Path = MODEL_DIR):
    """
    Charge le modèle entraîné.
    - source="hub"   : télécharge model.joblib depuis HF_MODEL_REPO
    - source="local" : charge depuis models/model.joblib
    """
    if source == "hub":
        logger.info("Téléchargement du modèle depuis %s", HF_MODEL_REPO)
        model_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="model.joblib",
            token=HF_TOKEN,
        )
    elif source == "local":
        model_file = model_path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} introuvable.")
    else:
        raise ValueError(f"source inconnue : {source!r} (attendu 'hub' ou 'local')")

    return joblib.load(model_file)


# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES FRAÎCHES
# ---------------------------------------------------------------------------
def load_input_data(input_path: Path = None) -> pd.DataFrame:
    """
    Charge les données fraîches à prédire.
    TODO: brancher la vraie source (API, base de données, fichier déposé
    périodiquement dans data/raw...). Le schéma attendu est le même que
    celui produit par src/data/make_dataset.py (sans la colonne 'cible').
    """
    if input_path is not None:
        return pd.read_csv(input_path)

    raise NotImplementedError(
        "Aucune source de données fraîches configurée — branche ici l'appel "
        "à ton API / base de données / fichier déposé périodiquement."
    )


# ---------------------------------------------------------------------------
# 3. PRÉDICTION
# ---------------------------------------------------------------------------
def predict(model, data: pd.DataFrame) -> pd.DataFrame:
    """Le pipeline sklearn complet gère feature engineering + preprocessing + prédiction."""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    result = data.copy()
    result["prediction"] = predictions
    result["probabilite"] = probabilities
    return result


def save_predictions(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génère des prédictions sur des données fraîches.")
    parser.add_argument("--source", choices=["local", "hub"], default="hub", help="Origine du modèle")
    parser.add_argument("--input", type=Path, default=None, help="Fichier de données fraîches à prédire")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fichier de sortie")
    args = parser.parse_args()

    model = load_model(source=args.source)
    data = load_input_data(args.input)
    results = predict(model, data)

    output_file = save_predictions(results, args.output)
    logger.info("Prédictions sauvegardées dans %s (%d lignes)", output_file, len(results))


if __name__ == "__main__":
    main()
