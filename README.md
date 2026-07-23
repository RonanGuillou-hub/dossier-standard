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

La suite couvre : `FeatureEngineer` (feature engineering), `make_dataset.py`
(nettoyage structurel, fusion météo), `train.py` (construction du pipeline,
entraînement, métriques), `predict_model.py` (chargement du modèle,
inférence), l'API FastAPI (tous les endpoints, météo mockée), et
`config.py`. Fixtures partagées dans `tests/conftest.py` — le modèle
d'entraînement (coûteux) n'est construit qu'une seule fois par session
de test (`trained_model`, scope `session`).

Les tests marqués `@pytest.mark.integration` (ex: `test_meteo.py`, vrai
appel à l'API météo) sont **exclus par défaut** (voir `pytest.ini`) pour
que la suite reste rapide et ne dépende pas du réseau :

```bash
pytest tests/                    # tests unitaires uniquement (rapide, hors ligne)
pytest tests/ -m integration      # tests d'intégration uniquement (réseau requis)
pytest tests/ -m ""                # tous les tests
```

## Notebook — exécution manuelle du pipeline

`notebooks/manual_pipeline_run.ipynb` déroule tout le pipeline étape par
étape (génération/nettoyage des données, météo, feature engineering,
entraînement, inférence, test de l'API), utile pour déboguer ou explorer
sans passer par le cron GitHub Actions ni HuggingFace Jobs. Si l'API météo
n'est pas accessible (pas de réseau), le notebook retombe automatiquement
sur des valeurs factices pour pouvoir continuer hors ligne.

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

1. **En local** : un fichier `.env` à la racine du projet (copier
   `.env.example`, jamais commité — voir `.gitignore`), chargé
   automatiquement par `src/config.py` (`python-dotenv`) dans
   `os.environ`. Une vraie variable d'environnement déjà définie n'est
   jamais écrasée par `.env` (`override=False`).
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
   modèle, logue les métriques vers MLflow, sauvegarde **toujours** le modèle
   dans `models/<model_filename>` (localement, indépendamment de HuggingFace),
   puis le pousse en plus vers le Hub **si `HF_TOKEN` est présent** — sinon un
   simple warning est loggé et le modèle reste disponible en local uniquement.

**Historisation des modèles entraînés** : chaque run sauvegarde deux
copies :
- `models/<model_filename>` (ex: `model.joblib`) — la version "courante",
  **écrasée à chaque run**, c'est celle que `predict_model.py` charge par
  défaut en `source="local"`
- `models/history/<model_filename_sans_extension>_<AAAAMMJJ_HHMMSS>.joblib`
  — une copie **datée, jamais écrasée**, pour retrouver n'importe quelle
  version entraînée précédemment

Si `HF_TOKEN` est présent, la copie historisée est aussi poussée sur le
Hub sous `history/<nom_du_fichier_daté>`, en plus de la version courante.

⚠️ Ni `models/` ni `models/history/` ne sont persistés au-delà d'un run sur
l'instance GPU HuggingFace (éphémère) — seul le push vers le Hub (ou S3, si
tu appliques le même mécanisme que pour `dataset_clean.csv`/`meteo.csv`)
survit à la fin du job.

**Erreur `PermissionError` sur `models/model.joblib` en local ?** Le plus
probable : ce fichier a été créé par un précédent `docker compose run`
(le conteneur tourne en root, donc le fichier appartient à `root` côté
host via le volume monté). Fix : `sudo chown -R $USER:$USER models/`.

**Chargement du modèle ailleurs (`FeatureEngineer` est une classe custom)** :
MLflow sérialise par défaut avec `skops` (plus sûr que `pickle`), qui bloque
tout type non explicitement listé comme fiable — `mlflow.sklearn.log_model()`
passe donc `skops_trusted_types=["src.models.train.FeatureEngineer", "numpy.dtype"]`
et `code_paths=["src", "configs"]` (le second car `train.py` lit
`config.yaml` au niveau module). Sans ces deux paramètres, le chargement du
modèle échoue ailleurs avec une erreur "untrusted types" ou `FileNotFoundError`.
Testé avec un vrai round-trip MLflow (log → load → predict) depuis un
répertoire totalement isolé du projet.

**Model Registry MLflow (obligatoire pour que l'API charge le modèle)** :
`train.py` enregistre le modèle sous `mlflow.registered_model_name`
(`config.yaml`) et lui assigne l'alias `mlflow.model_alias` (ex:
`@champion`) à chaque run — sans ça, l'API (`api.model_source: "mlflow"`)
échoue avec `RESOURCE_DOES_NOT_EXIST: Registered Model ... not found`.
Utilise des **alias**, pas les stages `Staging`/`Production` (dépréciés
par MLflow depuis la version 2.9).

⚠️ **Le Model Registry nécessite un backend base de données** côté
serveur MLflow (`sqlite:///...`, `postgresql://...`...) — un simple
stockage fichier (`file:///...` ou `./mlruns`) ne le supporte pas.
Si ton serveur MLflow tourne encore en mode fichier, `registered_model_name`
échouera silencieusement ou avec une erreur explicite selon la version.

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

## API d'inférence (FastAPI) + interface Streamlit

`src/api/` sert le modèle entraîné via une API REST. Contrairement à
`predict_model.py` (batch, déclenché par cron), l'API répond en temps réel
à des requêtes individuelles.

**Source du modèle** (`configs/config.yaml`, `api.model_source`) : `local`,
`s3` ou `mlflow` — interchangeable sans toucher au code.

### Lancer en local

```bash
make api          # démarre l'API sur http://localhost:8000 (docs interactives : /docs)
make streamlit     # démarre l'interface sur http://localhost:8501, dans un autre terminal
```

### Lancer avec Docker Compose

```bash
docker compose up api streamlit
```

L'API expose un `healthcheck` (`GET /health`), et `streamlit` attend
explicitement que l'API soit **prête** (`condition: service_healthy`)
avant de démarrer — pas juste que son conteneur ait été lancé. Sans ça,
`depends_on` seul garantit uniquement l'ordre de démarrage des
conteneurs, pas que le serveur `uvicorn` à l'intérieur ait fini de
charger le modèle : Streamlit pouvait alors afficher `Connection refused`
sur son premier appel, avant que l'API soit réellement joignable.

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Vérifie que l'API répond |
| GET | `/model/info` | Source et date de chargement du modèle en cache |
| POST | `/model/reload` | Force un rechargement depuis la source configurée |
| POST | `/predict` | Prédiction sur une observation unique |
| POST | `/predict/batch` | Prédiction sur plusieurs observations |

**Champs attendus par `/predict`** : les colonnes brutes du pipeline
(`age`, `revenu`, `anciennete_mois`, `categorie`, `region`) et,
optionnellement, `date` (`AAAA-MM-JJ`, par défaut aujourd'hui). L'API
récupère automatiquement la météo de la région/date via
`fetch_weather_for_region` (même fonction qu'à l'entraînement, pour
éviter toute divergence train/serving), avec un cache mémoire par
combinaison région/date pour éviter les appels redondants. Les colonnes
dérivées (`revenu_par_age`, `tranche_age`...) sont calculées
automatiquement par le pipeline (`FeatureEngineer`), inutile de les fournir.

⚠️ `fetch_weather_for_region` utilise l'API archive d'Open-Meteo, pensée
pour des dates passées — les données très récentes (derniers jours)
peuvent ne pas encore être disponibles. Pour de la météo du jour même en
production, il faudrait basculer vers l'API forecast d'Open-Meteo.

**Secrets requis selon `api.model_source`** :
- `mlflow` : `MLFLOW_TRACKING_URI`
- `s3` : `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `local` : aucun, lit `models/model.joblib`

## Prédictions automatisées (données fraîches, sans réentraînement)

Contrairement à l'entraînement, l'inférence avec ce modèle ne nécessite pas de
GPU : le pipeline complet (`.github/workflows/predict.yml`) tourne directement
sur le runner GitHub Actions (CPU), sans passer par HuggingFace Jobs.

1. **`.github/workflows/predict.yml`** : cron GitHub Actions, se déclenche
   périodiquement (par défaut toutes les heures) et exécute `predict_model.py`.
2. **`src/models/predict_model.py`** :
   - télécharge le modèle déjà entraîné depuis le Hub HuggingFace (`model.joblib`)
   - charge les données fraîches (source à brancher dans `load_input_data` —
     seul un fichier CSV via `--input` est supporté pour l'instant, pour
     tester manuellement)
   - **récupère automatiquement la météo** de chaque région/date via
     `enrich_with_weather()` (même logique que l'API, cache par combinaison
     région/date) — pas besoin de fournir `temperature_max/min`,
     `precipitation` dans le fichier d'entrée
   - génère les prédictions et les sauvegarde dans `reports/predictions.csv`
3. Les prédictions sont publiées comme **artefact GitHub Actions** (téléchargeable
   depuis l'onglet Actions, conservé 30 jours par défaut).

**Tester manuellement** avec l'exemple fourni :

```bash
make predict INPUT=data/examples/sample_input.csv
# ou directement, avec le choix de la source du modèle :
python -m src.models.predict_model --source local --input data/examples/sample_input.csv
python -m src.models.predict_model --source hub --input data/examples/sample_input.csv
python -m src.models.predict_model --source mlflow --input data/examples/sample_input.csv
```

`--source` accepte `local` (`models/<model_filename>`), `hub` (Hub
HuggingFace, `HF_MODEL_REPO`) ou `mlflow` (Model Registry MLflow, via
`mlflow.registered_model_name`/`model_alias` — même mécanisme que l'API,
voir `src/api/model_loader.py`). Par défaut : `hub`.

`data/examples/sample_input.csv` contient uniquement les colonnes brutes
(`age`, `revenu`, `anciennete_mois`, `categorie`, `region`) — un `date`
optionnel peut être ajouté par ligne (`AAAA-MM-JJ`, défaut : aujourd'hui).

**Variables/secrets requis dans le repo GitHub :**

- `HF_TOKEN` (secret) : même token que pour l'entraînement
- `HF_MODEL_REPO` (variable, `Settings > Secrets and variables > Actions > Variables`) :
  nom du repo HuggingFace où le modèle entraîné est stocké

**À compléter avant utilisation en production :**

- `load_input_data()` dans `predict_model.py` : brancher la vraie source de
  données fraîches (API, base de données, fichier déposé périodiquement...)
  — actuellement, seul `--input <fichier.csv>` fonctionne (mode test/manuel)
