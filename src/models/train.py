"""
Script d'entraînement réel : exécuté SUR l'instance GPU HuggingFace Jobs
(déclenchée par trigger_job.py). C'est ici que le tracking MLflow
et le push du modèle vers le Hub ont lieu.
"""

import os

import mlflow
import mlflow.pytorch
from huggingface_hub import HfApi

# --- Configuration MLflow ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://ton-serveur-mlflow.com")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "mon-projet-ml")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "mon-user/mon-modele")
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_data():
    """TODO: charger/prétraiter les données (depuis data/processed par ex.)."""
    raise NotImplementedError


def build_model():
    """TODO: instancier le modèle."""
    raise NotImplementedError


def train(model, data):
    """TODO: boucle d'entraînement, retourne le modèle entraîné + métriques."""
    raise NotImplementedError


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        params = {
            "learning_rate": 1e-4,
            "epochs": 10,
            "batch_size": 32,
        }
        mlflow.log_params(params)

        data = load_data()
        model = build_model()
        trained_model, metrics = train(model, data)

        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(trained_model, "model")

        # --- Push du modèle final vers le Hub HuggingFace ---
        if HF_TOKEN:
            api = HfApi(token=HF_TOKEN)
            trained_model.save_pretrained("outputs/model")
            api.upload_folder(
                folder_path="outputs/model",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print(f"Modèle poussé vers {HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
