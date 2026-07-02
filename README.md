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
python -m src.models.train_model
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
# dossier-standard
