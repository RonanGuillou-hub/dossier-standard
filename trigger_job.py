"""
Script déclencheur : tourne sur le runner GitHub Actions (CPU, pas de GPU).
Il ne fait AUCUN entraînement lui-même — il demande à HuggingFace
de lancer, sur une instance GPU à la demande :
  1. src/data/make_dataset.py (génère data/processed/dataset_clean.csv,
     y compris la fusion météo et son upload S3 — nécessaire car
     l'instance GPU est éphémère et ne conserve rien après le job)
  2. src/models/train.py (entraîne le modèle à partir de ce fichier)

Important : l'instance GPU est un environnement totalement séparé du
runner GitHub Actions. Les variables d'environnement de ce dernier ne
sont PAS automatiquement transmises — il faut les passer explicitement
via les paramètres env= (non-secret) et secrets= (secret, chiffré côté
serveur HuggingFace) de run_job().
"""

import os

from huggingface_hub import run_job

from src.config import load_config

CONFIG = load_config()
HF_TOKEN = os.environ["HF_TOKEN"]

# Variables non-secrètes à transmettre à l'instance GPU. Valeur du
# runner GitHub Actions si définie, sinon repli sur config.yaml.
# `or default` (et non juste `.get(key, default)`) car une variable
# GitHub Actions non configurée arrive comme chaîne vide, pas absente.
job_env = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI") or CONFIG["mlflow"]["default_tracking_uri"],
    "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME") or CONFIG["mlflow"]["experiment_name"],
    "HF_MODEL_REPO": os.environ.get("HF_MODEL_REPO") or CONFIG["huggingface"]["model_repo"],
}

# Secrets chiffrés côté serveur HuggingFace. AWS_* est optionnel : si
# absent, l'upload S3 du fichier météo est simplement ignoré (voir
# upload_meteo_to_s3 dans src/data/make_dataset.py).
job_secrets = {"HF_TOKEN": HF_TOKEN}
if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
    job_secrets["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID"]
    job_secrets["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY"]

job = run_job(
    image=CONFIG["huggingface"]["job_image"],
    command=["bash", "-c", "python -m src.data.make_dataset && python -m src.models.train"],
    flavor=CONFIG["huggingface"]["job_flavor"],
    repo_id=CONFIG["huggingface"]["job_repo"],
    env=job_env,
    secrets=job_secrets,
    token=HF_TOKEN,
)

print(f"Job lancé : {job}")
print(f"Variables d'environnement transmises : {list(job_env.keys())}")
print(f"Secrets transmis : {list(job_secrets.keys())}")
