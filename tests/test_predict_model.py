"""Tests de src/models/predict_model.py — chargement du modèle et prédiction."""

import joblib
import pytest

from src.models.predict_model import load_model, predict


def test_load_model_local_leve_erreur_si_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(source="local", model_path=tmp_path)


def test_load_model_local_charge_correctement(tmp_path, trained_model):
    model_file = tmp_path / "model.joblib"
    joblib.dump(trained_model, model_file)

    loaded = load_model(source="local", model_path=tmp_path)
    assert hasattr(loaded, "predict")


def test_load_model_source_invalide_leve_value_error():
    with pytest.raises(ValueError, match="source inconnue"):
        load_model(source="invalide")


def test_predict_ajoute_prediction_et_probabilite(trained_model, X_y):
    X, _ = X_y
    result = predict(trained_model, X.head(3))

    assert "prediction" in result.columns
    assert "probabilite" in result.columns
    assert len(result) == 3
    assert result["probabilite"].between(0, 1).all()


def test_predict_conserve_les_colonnes_dorigine(trained_model, X_y):
    X, _ = X_y
    result = predict(trained_model, X.head(3))
    for col in X.columns:
        assert col in result.columns
