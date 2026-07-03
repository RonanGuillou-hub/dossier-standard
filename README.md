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

## Configuration (configs/config.yaml)

Tous les paramètres **non-secrets** du projet sont centralisés dans
`configs/config.yaml` : chemins, colonnes du modèle, hyperparamètres,
paramètres d'entraînement, régions/coordonnées météo, noms de repos
HuggingFace, nom d'expérience MLflow.

Chargement via `src/config.py` :

```python
from src.config import load_config
config = load_config()
```

**Ce qui reste en variable d'environnement (jamais dans ce fichier) :**

- `HF_TOKEN` : token HuggingFace
- `MLFLOW_TRACKING_URI` : URL du serveur MLflow (une valeur par défaut
  non-secrète existe dans `config.yaml`, mais la variable d'environnement
  prend toujours le dessus si elle est définie)
- `HF_MODEL_REPO` : peut aussi être surchargé via variable d'environnement
  si tu veux pointer temporairement vers un autre repo sans toucher au YAML

Pour changer un hyperparamètre (ex: `test_size`, `max_iter`) ou ajouter une
région météo, il suffit d'éditer `configs/config.yaml` — aucun code à
toucher.

### Modifier MLFLOW_TRACKING_URI (ou HF_MODEL_REPO, HF_TOKEN)

Ces valeurs vivent à 3 endroits différents, à mettre à jour séparément
selon où le code s'exécute :

1. **En local** : `export MLFLOW_TRACKING_URI=https://...` dans ton shell,
   ou un fichier `.env` chargé avec `python-dotenv` (non fourni par défaut).
2. **GitHub Actions** (`trigger_job.py`, déclenché par le cron) :
   `Settings > Secrets and variables > Actions > Variables` → ajouter
   `MLFLOW_TRACKING_URI`. Le workflow `train.yml` l'expose déjà au step
   via `${{ vars.MLFLOW_TRACKING_URI }}`.
3. **Instance GPU HuggingFace** (là où `train.py` s'exécute réellement) :
   **ne se configure jamais directement** — `trigger_job.py` doit la
   transmettre explicitement via le paramètre `env=` de `run_job()`, car
   l'instance GPU est un environnement totalement séparé du runner
   GitHub Actions. C'est déjà câblé dans `trigger_job.py` ; si tu ajoutes
   une nouvelle variable d'environnement au projet, pense à l'ajouter
   aussi à `job_env` dans ce fichier, sinon elle n'atteindra jamais
   `train.py`.

Sans l'étape 3, la variable existerait bien sur le runner GitHub mais
`train.py` ne la verrait jamais et retomberait silencieusement sur la
valeur par défaut de `config.yaml`.

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

## Données externes (météo)

`src/data/make_dataset.py` enrichit le dataset avec des données météo via
l'API publique [Open-Meteo](https://open-meteo.com/) (gratuite, sans clé),
appelée à **chaque exécution** — pas de cache utilisé comme source de vérité.
La fusion se fait sur les clés `(date, région)`.

- `REGION_COORDINATES` : mapping région → coordonnées, à adapter à ton
  vrai découpage géographique
- `data/external/meteo.csv` : sauvegarde de la dernière réponse API, à
  titre de trace/inspection uniquement (régénéré à chaque run)
- `--skip-external` : ignore l'appel API météo, pour tester rapidement
  le nettoyage seul, hors ligne. **Ne pas utiliser en production** : le
  modèle attend les colonnes météo et échouera sans elles.

Si tu changes de fournisseur météo (clé API requise, autre région...),
adapte `fetch_weather_for_region()` en conséquence.

### Persistance sur S3

L'instance GPU HuggingFace où `make_dataset.py` s'exécute est **éphémère** :
tout ce qui n'est pas explicitement sauvegardé ailleurs disparaît à la fin
du job — y compris `data/external/meteo.csv` et `data/processed/dataset_clean.csv`,
qui ne sont ni commités sur GitHub (`.gitignore`) ni stockés nulle part
par défaut.

Un seul bucket S3 partagé (`configs/config.yaml`, section `s3`) reçoit les
deux fichiers, chacun sous son propre préfixe. Structure de clé :

```
{prefix}/{année}/{mois}/{nom}_{AAAAMMJJ}.csv

# Exemples concrets :
processed/dataset_clean/2026/07/dataset_clean_20260703.csv
external/meteo/2026/07/meteo_20260703.csv
```

⚠️ La clé ne contient que la date, pas l'heure : plusieurs runs le même
jour écrasent le même fichier S3. Si tu veux un fichier distinct par run,
ajoute l'heure dans `upload_file_to_s3()` (`src/data/make_dataset.py`).

**Secrets requis** (jamais dans `config.yaml`) :
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — dans les GitHub Secrets
  du repo, transmis à l'instance GPU par `trigger_job.py`

Si ces credentials sont absents (ex: développement local sans accès AWS),
l'upload est simplement ignoré avec un warning — pas d'erreur bloquante.

## Prédictions automatisées (données fraîches, sans réentraînement)

Contrairement à l'entraînement, l'inférence avec ce modèle ne nécessite pas de
GPU : le pipeline complet (`.github/workflows/predict.yml`) tourne directement
sur le runner GitHub Actions (CPU), sans passer par HuggingFace Jobs.

1. **`.github/workflows/predict.yml`** : cron GitHub Actions, se déclenche
   périodiquement (par défaut toutes les heures) et exécute `predict_model.py`.
2. **`src/models/predict_model.py`** :
   - télécharge le modèle déjà entraîné depuis le Hub HuggingFace (`model.joblib`)
   - charge les données fraîches (source à brancher dans `load_input_data`)
   - génère les prédictions et les sauvegarde dans `reports/predictions.csv`
3. Les prédictions sont publiées comme **artefact GitHub Actions** (téléchargeable
   depuis l'onglet Actions, conservé 30 jours par défaut).

**Variables/secrets requis dans le repo GitHub :**

- `HF_TOKEN` (secret) : même token que pour l'entraînement
- `HF_MODEL_REPO` (variable, `Settings > Secrets and variables > Actions > Variables`) :
  nom du repo HuggingFace où le modèle entraîné est stocké

**À compléter avant utilisation :**

- `load_input_data()` dans `predict_model.py` : brancher la vraie source de
  données fraîches (API, base de données, fichier déposé périodiquement...)
