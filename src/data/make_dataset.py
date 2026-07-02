"""
Chargement et nettoyage structurel des données.

Transforme les données brutes (data/raw) en données prêtes à l'emploi
pour le feature engineering / l'entraînement (data/processed).

Le nettoyage effectué ici est STRUCTUREL uniquement : il ne dépend
d'aucune statistique apprise (pas de moyenne, médiane, encodage...) et
ne crée donc aucune fuite de données entre train et test. C'est pour
cette raison qu'il vit ici, hors du pipeline sklearn, et s'applique
une seule fois sur l'ensemble du dataset avant le split train/test.

Les transformations qui, elles, dépendent de statistiques apprises sur
le train (imputation, scaling, encodage) vivent dans src/models/train.py,
à l'intérieur du pipeline sklearn.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

REQUIRED_COLUMNS = ["age", "revenu", "anciennete_mois", "categorie", "region", "cible"]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. GÉNÉRATION D'UN DATASET SYNTHÉTIQUE "SALE" (à remplacer par un vrai chargement)
# ---------------------------------------------------------------------------
def generate_dirty_dataset(n: int = 500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Crée un dataset avec des défauts typiques : NaN, doublons, valeurs
    aberrantes. Utilisé ici à des fins de démonstration/développement,
    tant qu'aucune vraie source de données n'est branchée.
    """
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(1),
        "revenu": rng.normal(35000, 9000, n).round(0),
        "anciennete_mois": rng.integers(0, 240, n),
        "categorie": rng.choice(["A", "B", "C", None], size=n, p=[0.4, 0.3, 0.25, 0.05]),
        "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], size=n),
        "cible": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
    })

    # Injection de valeurs manquantes réalistes
    for col in ["age", "revenu", "anciennete_mois"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    # Valeurs aberrantes évidentes (erreurs de saisie)
    df.loc[rng.choice(n, 3, replace=False), "age"] = -5
    df.loc[rng.choice(n, 3, replace=False), "revenu"] = 999999999

    # Doublons volontaires
    df = pd.concat([df, df.sample(10, random_state=random_state)], ignore_index=True)

    return df


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Charge les données brutes.

    Pour l'instant, génère un dataset synthétique tant qu'aucune vraie
    source n'est branchée. À remplacer par un `pd.read_csv(input_path / "...")`,
    une requête SQL, un appel API, etc. selon le projet.
    """
    logger.info("Chargement des données brutes depuis %s (dataset synthétique)", input_path)
    return generate_dirty_dataset()


# ---------------------------------------------------------------------------
# 2. NETTOYAGE STRUCTUREL (avant le split, ne dépend d'aucune statistique apprise)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrections qui ne "voient" pas la cible et ne créent pas de fuite de
    données :
    - validation du schéma attendu
    - suppression des doublons
    - correction des valeurs aberrantes évidentes (bornes physiques connues)
    - typage correct des colonnes
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset brut : {missing}")

    df = df.copy()

    # a) Déduplication
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("%d doublons supprimés", n_before - len(df))

    # b) Valeurs aberrantes évidentes -> remplacées par NaN (imputées plus tard, dans le pipeline)
    df.loc[df["age"] < 0, "age"] = np.nan
    df.loc[df["age"] > 110, "age"] = np.nan
    df.loc[df["revenu"] > 500000, "revenu"] = np.nan

    # c) Typage explicite
    df["categorie"] = df["categorie"].astype("category")
    df["region"] = df["region"].astype("category")

    try:
        df["cible"] = df["cible"].astype(int)
    except ValueError as e:
        raise ValueError(
            "Impossible de convertir 'cible' en int — vérifie s'il reste des NaN "
            "ou des valeurs non numériques dans cette colonne."
        ) from e

    # d) Reset index propre
    df = df.reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> Path:
    """Sauvegarde le dataset nettoyé au format CSV."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "dataset_clean.csv"
    df.to_csv(output_file, index=False)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset pour l'entraînement.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR, help="Dossier des données brutes")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR, help="Dossier de sortie")
    args = parser.parse_args()

    df = load_raw_data(args.input)
    df = clean_data(df)
    output_file = save_processed_data(df, args.output)

    logger.info("Données traitées sauvegardées dans %s (%d lignes)", output_file, len(df))


if __name__ == "__main__":
    main()
