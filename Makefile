.PHONY: install data train predict test lint clean docker-build docker-train docker-notebook

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
