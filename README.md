# Mon Projet ML

Description courte du projet.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Structure du projet

```
mon-projet-ml/
├── data/            # Données (raw, interim, processed, external)
├── notebooks/       # Notebooks d'exploration
├── src/             # Code source (data, features, models, evaluation, visualization)
├── models/          # Modèles entraînés
├── reports/         # Rapports et figures générés
├── configs/         # Fichiers de configuration
└── tests/           # Tests unitaires
```

## Usage

```bash
python -m src.data.make_dataset
python -m src.models.train
python -m src.models.predict_model
```

## Tests

```bash
pytest tests/
```

## Docker

Construire l'image et lancer l'entraînement dans un conteneur :

```bash
docker compose build
docker compose run --rm ml
```

Lancer un Jupyter Notebook accessible sur `http://localhost:8888` :

```bash
docker compose up notebook
```

## Entraînement automatisé (GPU à la demande + MLflow + Hub)

Le pipeline complet fonctionne en 3 étapes :

1. **`.github/workflows/train.yml`** : cron GitHub Actions (aucun GPU), se déclenche
   périodiquement et appelle `trigger_job.py`.
2. **`trigger_job.py`** : demande à HuggingFace Jobs de lancer une instance GPU
   à la demande et d'y exécuter `src/models/train.py`.
3. **`src/models/train.py`** : tourne SUR l'instance GPU HuggingFace. Il entraîne le
   modèle, logue les métriques vers MLflow, et pousse le modèle final sur le Hub.

**Secrets requis dans le repo GitHub** (`Settings > Secrets and variables > Actions`) :

- `HF_TOKEN` : token HuggingFace avec droits d'écriture sur le repo de job et le modèle

**Variables d'environnement à définir côté HuggingFace Jobs** (dans `trigger_job.py`
ou via la configuration du job) :

- `MLFLOW_TRACKING_URI` : URL de ton serveur MLflow
- `MLFLOW_EXPERIMENT_NAME` : nom de l'expérience MLflow
- `HF_MODEL_REPO` : repo HuggingFace où pousser le modèle entraîné
- `HF_TOKEN` : même token que ci-dessus, transmis au job

Pour tester manuellement le déclenchement sans attendre le cron, va dans l'onglet
**Actions** du repo GitHub et lance le workflow via *"Run workflow"*.
