"""
Script d'entraînement réel : exécuté SUR l'instance GPU HuggingFace Jobs
(déclenchée par trigger_job.py).

Contenu :
- FeatureEngineer : transformer sklearn custom (feature engineering),
  fait partie du pipeline car doit rester identique train/test/prod.
- ColumnTransformer : imputation + scaling + encodage, fit UNIQUEMENT
  sur le train (aucune fuite de données).
- Pipeline complet : feature engineering + preprocessing + modèle.
- Tracking MLflow (paramètres, métriques, modèle) + push du modèle
  entraîné vers le Hub HuggingFace.

Le chargement des données brutes et leur nettoyage structurel
(déduplication, correction des valeurs aberrantes évidentes...) ont
déjà eu lieu en amont, dans src/data/make_dataset.py. Ce script part
directement de data/processed/dataset_clean.csv.
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

PROCESSED_DATA_PATH = Path("data/processed/dataset_clean.csv")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://ton-serveur-mlflow.com")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "mon-projet-ml")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "mon-user/mon-modele")
HF_TOKEN = os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING (dans le pipeline, car doit rester identique train/test/prod)
# ---------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn custom qui crée de nouvelles colonnes à partir des
    colonnes brutes (déjà nettoyées structurellement, mais pouvant encore
    contenir des NaN -> ces NaN se propagent naturellement dans les nouvelles
    colonnes et seront traités par l'imputer du ColumnTransformer en aval).

    Placé dans le pipeline (et non dans make_dataset.py) car :
    - une future feature pourrait dépendre d'une statistique du train
      (ex: écart à la moyenne du groupe) et devrait alors être fit sur
      train uniquement ;
    - on veut que la même logique s'applique automatiquement à toute
      nouvelle donnée passée à `predict()`, sans étape manuelle séparée.
    """

    def fit(self, X, y=None):
        # Stateless ici (aucune statistique apprise), mais fit() doit exister
        # et renvoyer self pour être compatible avec l'API sklearn.
        return self

    def transform(self, X):
        X = X.copy()

        # a) Feature ratio : revenu par année d'ancienneté
        anciennete_annees = X["anciennete_mois"] / 12
        X["revenu_par_annee_anciennete"] = X["revenu"] / anciennete_annees.replace(0, np.nan)

        # b) Feature ratio : revenu par âge (proxy de "revenu précoce")
        X["revenu_par_age"] = X["revenu"] / X["age"].replace(0, np.nan)

        # c) Feature de bucket : tranche d'âge (catégorielle dérivée d'une numérique)
        X["tranche_age"] = pd.cut(
            X["age"],
            bins=[0, 25, 40, 55, 120],
            labels=["18-25", "26-40", "41-55", "56+"],
        ).astype("object")  # object pour laisser l'imputer catégoriel gérer les NaN

        # d) Feature d'interaction : combinaison categorie x region
        X["categorie_region"] = (
            X["categorie"].astype(str) + "_" + X["region"].astype(str)
        )

        # e) Remplacement des divisions par zéro / infinis générés en (a) et (b)
        X = X.replace([np.inf, -np.inf], np.nan)

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + [
            "revenu_par_annee_anciennete",
            "revenu_par_age",
            "tranche_age",
            "categorie_region",
        ]


NUMERIC_COLS = [
    "age", "revenu", "anciennete_mois",
    "revenu_par_annee_anciennete", "revenu_par_age",
]
CATEGORICAL_COLS = [
    "categorie", "region",
    "tranche_age", "categorie_region",
]


def build_model() -> Pipeline:
    """Construit le pipeline complet : feature engineering + preprocessing + modèle."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])

    return Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES PRÉTRAITÉES (produites par src/data/make_dataset.py)
# ---------------------------------------------------------------------------
def load_data(path: Path = PROCESSED_DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord `python -m src.data.make_dataset` "
            "pour générer les données prétraitées."
        )
    df = pd.read_csv(path)
    X = df.drop(columns=["cible"])
    y = df["cible"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 3. ENTRAÎNEMENT + ÉVALUATION + VALIDATION CROISÉE
# ---------------------------------------------------------------------------
def train(model: Pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Accuracy (test) : %.3f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    logger.info("Accuracy (CV, 5 folds) : %.3f (+/- %.3f)", cv_scores.mean(), cv_scores.std())

    metrics = {
        "accuracy": accuracy,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "precision_classe_1": report["1"]["precision"],
        "recall_classe_1": report["1"]["recall"],
        "f1_classe_1": report["1"]["f1-score"],
    }

    return model, metrics


# ---------------------------------------------------------------------------
# 4. MAIN : tracking MLflow + push du modèle vers le Hub HuggingFace
# ---------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        try:
            X, y = load_data()
            model = build_model()

            params = {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
            }
            mlflow.log_params(params)

            trained_model, metrics = train(model, X, y)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(trained_model, "model")

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            logger.exception("Échec de l'entraînement")
            raise

        # --- Push du modèle final vers le Hub HuggingFace ---
        if HF_TOKEN:
            import joblib

            output_dir = Path("outputs/model")
            output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(trained_model, output_dir / "model.joblib")

            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            logger.info("Modèle poussé vers %s", HF_MODEL_REPO)
        else:
            logger.warning("HF_TOKEN absent : le modèle n'a pas été poussé sur le Hub.")


if __name__ == "__main__":
    main()
