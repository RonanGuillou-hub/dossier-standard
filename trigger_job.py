"""
Script déclencheur : tourne sur le runner GitHub Actions (CPU, pas de GPU).
Il ne fait AUCUN entraînement lui-même — il demande à HuggingFace
de lancer, sur une instance GPU à la demande :
  1. src/data/make_dataset.py (génère data/processed/dataset_clean.csv,
     y compris la fusion météo — nécessaire car l'instance GPU est
     éphémère et ne contient aucune donnée pré-générée)
  2. src/models/train.py (entraîne le modèle à partir de ce fichier)
"""

import os

from huggingface_hub import run_job

HF_TOKEN = os.environ["HF_TOKEN"]

# Adapter le nom de repo / image / flavor selon ton besoin.
# Liste des "flavors" GPU disponibles : https://huggingface.co/docs/hub/spaces-gpus
job = run_job(
    image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    command=["bash", "-c", "python -m src.data.make_dataset && python -m src.models.train"],
    flavor="a10g-small",
    repo_id="mon-user/mon-repo-job",  # dépôt HF où le job s'exécute
    token=HF_TOKEN,
)

print(f"Job lancé : {job}")
