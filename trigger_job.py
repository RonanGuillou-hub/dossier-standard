"""
Script déclencheur : tourne sur le runner GitHub Actions (CPU, pas de GPU).
Il ne fait AUCUN entraînement lui-même — il demande à HuggingFace
de lancer `src/models/train.py` sur une instance GPU à la demande.
"""

import os

from huggingface_hub import run_job

HF_TOKEN = os.environ["HF_TOKEN"]

# Adapter le nom de repo / image / flavor selon ton besoin.
# Liste des "flavors" GPU disponibles : https://huggingface.co/docs/hub/spaces-gpus
job = run_job(
    image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    command=["python", "src/models/train.py"],
    flavor="a10g-small",
    repo_id="mon-user/mon-repo-job",  # dépôt HF où le job s'exécute
    token=HF_TOKEN,
)

print(f"Job lancé : {job}")
