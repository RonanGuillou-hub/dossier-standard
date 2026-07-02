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
│
├── README.md                  # Présentation du projet, installation, usage
├── .gitignore                 # Exclure data brute, venv, checkpoints, etc.
├── requirements.txt           # ou environment.yml / pyproject.toml
├── setup.py                   # si le projet est packagé
├── Makefile                   # commandes courantes (train, test, lint...)
│
├── data/
│   ├── raw/                   # Données brutes, jamais modifiées
│   ├── interim/                # Données intermédiaires (nettoyage partiel)
│   ├── processed/             # Données prêtes pour l'entraînement
│   └── external/               # Données provenant de sources tierces
│
├── notebooks/                 # Jupyter notebooks (exploration, EDA)
│   └── 01_exploration.ipynb
│
├── src/                        # Code source du projet
│   ├── __init__.py
│   ├── data/                   # Scripts de chargement / prétraitement
│   │   ├── make_dataset.py
│   │   └── preprocessing.py
│   ├── features/               # Feature engineering
│   │   └── build_features.py
│   ├── models/                 # Entraînement, prédiction
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── evaluation/              # Métriques, validation
│   │   └── evaluate.py
│   └── visualization/           # Génération de graphiques
│       └── visualize.py
│
├── models/                     # Modèles entraînés sauvegardés (souvent ignoré par git)
│   └── model_v1.pkl
│
├── reports/                    # Résultats, figures, rapports
│   ├── figures/
│   └── final_report.md
│
├── configs/                    # Fichiers de configuration (YAML/JSON)
│   └── config.yaml
│
├── tests/                      # Tests unitaires
│   └── test_preprocessing.py
│
└── docs/                       # Documentation complémentaire
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
