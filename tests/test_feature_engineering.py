"""Tests du transformer FeatureEngineer (src/models/feature_engineering.py)."""

import numpy as np
import pandas as pd
import pytest

from src.models.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [30, 45, np.nan, 0],
        "revenu": [30000, 50000, 40000, 20000],
        "anciennete_mois": [12, 0, 24, 36],
        "categorie": ["A", "B", "C", "A"],
        "region": ["Nord", "Sud", "Est", "Ouest"],
    })


def test_ajoute_les_colonnes_derivees(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    for col in ["revenu_par_annee_anciennete", "revenu_par_age", "tranche_age", "categorie_region"]:
        assert col in result.columns


def test_gere_division_par_zero_sans_produire_dinf(sample_df):
    """anciennete_mois=0 (ligne 2) et age=0 (ligne 4) ne doivent jamais produire d'inf."""
    result = FeatureEngineer().fit_transform(sample_df)
    assert not np.isinf(result["revenu_par_annee_anciennete"]).any()
    assert not np.isinf(result["revenu_par_age"]).any()


def test_categorie_region_est_une_concatenation(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert result.loc[0, "categorie_region"] == "A_Nord"


def test_tranche_age_respecte_les_bornes(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert result.loc[0, "tranche_age"] == "26-40"  # age=30
    assert result.loc[1, "tranche_age"] == "41-55"  # age=45


def test_fit_retourne_self(sample_df):
    fe = FeatureEngineer()
    assert fe.fit(sample_df) is fe


def test_ne_mute_pas_le_dataframe_original(sample_df):
    original_columns = list(sample_df.columns)
    FeatureEngineer().transform(sample_df)
    assert list(sample_df.columns) == original_columns


def test_get_feature_names_out_inclut_les_colonnes_derivees():
    names = FeatureEngineer().get_feature_names_out(["age", "revenu"])
    assert names == [
        "age", "revenu",
        "revenu_par_annee_anciennete", "revenu_par_age",
        "tranche_age", "categorie_region",
    ]
