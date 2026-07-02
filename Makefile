.PHONY: install data train predict test lint clean docker-build docker-train docker-notebook

# pour lancer à la main des commandes, dans le bash : make +NomDuPointEntree
# make install 			: Installe les dépendances Python (pip install -r requirements.txt)
# make data 			: Lance src/data/make_dataset.py (génération + nettoyage)
# make train			: Lance src/models/train.py (entraînement local)
# make predict			: Lance src/models/predict_model.py (inférence)
# make test				: Lance pytest tests/
# make lint				: Vérifie le style du code avec flake8
# make clean			: Supprime les dossiers __pycache__
# make docker-build 	: Construit l'image Docker (docker compose build)
# make docker-train		: Lance l'entraînement dans un conteneur
# make docker-notebook 	:Lance Jupyter dans un conteneur

# note : 
# Si tu modifies le Makefile toi-même et que make te répond Makefile:12: *** missing separator. Stop., 
# c'est presque toujours parce qu'un éditeur a remplacé la tabulation par des espaces — 
# c'est la source de bugs #1 avec les Makefiles.

install:
	pip install -r requirements.txt

docker-build:
	docker compose build

docker-train:
	docker compose run --rm ml

docker-notebook:
	docker compose up notebook

data:
	python -m src.data.make_dataset

train:
	python -m src.models.train

predict:
	python -m src.models.predict_model

test:
	pytest tests/

lint:
	flake8 src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
