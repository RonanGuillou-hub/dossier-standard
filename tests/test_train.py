"""Tests de src/models/train.py — construction du pipeline et entraînement."""

import pytest
from sklearn.pipeline import Pipeline

from src.models.train import build_model, load_data, train


def test_build_model_retourne_un_pipeline_avec_les_bonnes_etapes():
    model = build_model()
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["feature_engineering", "preprocessing", "classifier"]


def test_load_data_leve_erreur_explicite_si_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_dataset"):
        load_data(path=tmp_path / "inexistant.csv")


def test_train_retourne_un_modele_et_des_metriques_completes(X_y):
    X, y = X_y
    model = build_model()
    trained_model, metrics = train(model, X, y)

    assert hasattr(trained_model, "predict")
    for key in [
        "accuracy", "cv_accuracy_mean", "cv_accuracy_std",
        "precision_classe_1", "recall_classe_1", "f1_classe_1",
    ]:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_modele_entraine_peut_predire(trained_model, X_y):
    X, _ = X_y
    predictions = trained_model.predict(X.head(5))
    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})


def test_modele_entraine_expose_predict_proba(trained_model, X_y):
    X, _ = X_y
    probabilities = trained_model.predict_proba(X.head(5))
    assert probabilities.shape == (5, 2)
    # Chaque ligne de probabilités doit sommer à 1
    assert probabilities.sum(axis=1) == pytest.approx(1.0)
