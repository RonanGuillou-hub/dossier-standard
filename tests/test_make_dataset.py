"""Tests de src/data/make_dataset.py — nettoyage structurel et fusion météo."""

import pandas as pd
import pytest

from src.data.make_dataset import clean_data, merge_external_data


def _base_row(**overrides):
    row = {
        "date": pd.Timestamp("2026-01-01"),
        "age": 30,
        "revenu": 30000,
        "anciennete_mois": 12,
        "categorie": "A",
        "region": "Nord",
        "cible": 0,
    }
    row.update(overrides)
    return row


def test_clean_data_supprime_les_doublons():
    df = pd.DataFrame([_base_row(), _base_row()])
    result = clean_data(df)
    assert len(result) == 1


def test_clean_data_corrige_age_negatif():
    df = pd.DataFrame([_base_row(age=-5)])
    result = clean_data(df)
    assert pd.isna(result.loc[0, "age"])


def test_clean_data_corrige_age_superieur_a_110():
    df = pd.DataFrame([_base_row(age=150)])
    result = clean_data(df)
    assert pd.isna(result.loc[0, "age"])


def test_clean_data_corrige_revenu_aberrant():
    df = pd.DataFrame([_base_row(revenu=999999999)])
    result = clean_data(df)
    assert pd.isna(result.loc[0, "revenu"])


def test_clean_data_leve_erreur_si_colonne_manquante():
    df = pd.DataFrame([{"age": 30}])
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        clean_data(df)


def test_clean_data_type_cible_en_int():
    df = pd.DataFrame([_base_row(cible=1)])
    result = clean_data(df)
    assert result["cible"].dtype.kind == "i"


def test_merge_external_data_fusionne_sur_date_et_region():
    df = pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "region": "Nord", "age": 30}])
    external_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-01-01"), "region": "Nord",
        "temperature_max": 15.0, "temperature_min": 5.0, "precipitation": 1.2,
    }])
    result = merge_external_data(df, external_df)
    assert result.loc[0, "temperature_max"] == 15.0


def test_merge_external_data_signale_les_lignes_sans_correspondance(caplog):
    df = pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "region": "Nord", "age": 30}])
    external_df = pd.DataFrame([{
        "date": pd.Timestamp("2099-01-01"), "region": "Nord",  # ne correspond à rien
        "temperature_max": 15.0, "temperature_min": 5.0, "precipitation": 1.2,
    }])
    result = merge_external_data(df, external_df)
    assert pd.isna(result.loc[0, "temperature_max"])
