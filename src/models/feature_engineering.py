"""
FeatureEngineer — isolé dans son propre fichier, jamais exécuté comme
script (toujours importé), pour garantir un __module__ stable :
"src.models.feature_engineering.FeatureEngineer" en toutes circonstances.

Pourquoi c'est nécessaire : `python -m src.models.train` (utilisé par le
Makefile, le Dockerfile, trigger_job.py) donne à train.py un __name__
de "__main__" — EXACTEMENT comme `python src/models/train.py` lancé
directement. Si FeatureEngineer était définie dans train.py, son
__module__ deviendrait "__main__" dans ce cas, ce qui casse
`skops_trusted_types` lors du log_model MLflow (voir train.py) : le
nom qualifié utilisé au moment de sauvegarder le modèle ne correspondrait
à rien d'important ni de stable pour le chargement ailleurs.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn custom qui crée de nouvelles colonnes à partir des
    colonnes brutes (déjà nettoyées structurellement, mais pouvant encore
    contenir des NaN -> ces NaN se propagent naturellement dans les nouvelles
    colonnes et seront traités par l'imputer du ColumnTransformer en aval).

    Placé dans le pipeline (et non dans make_dataset.py) car :
    - une future feature pourrait dépendre d'une statistique du train
      (ex: écart à la moyenne du groupe) et devrait alors être fit sur
      train uniquement ;
    - on veut que la même logique s'applique automatiquement à toute
      nouvelle donnée passée à `predict()`, sans étape manuelle séparée.
    """

    def fit(self, X, y=None):
        # Stateless ici (aucune statistique apprise), mais fit() doit exister
        # et renvoyer self pour être compatible avec l'API sklearn.
        return self

    def transform(self, X):
        X = X.copy()

        # a) Feature ratio : revenu par année d'ancienneté
        anciennete_annees = X["anciennete_mois"] / 12
        X["revenu_par_annee_anciennete"] = X["revenu"] / anciennete_annees.replace(0, np.nan)

        # b) Feature ratio : revenu par âge (proxy de "revenu précoce")
        X["revenu_par_age"] = X["revenu"] / X["age"].replace(0, np.nan)

        # c) Feature de bucket : tranche d'âge (catégorielle dérivée d'une numérique)
        X["tranche_age"] = pd.cut(
            X["age"],
            bins=[0, 25, 40, 55, 120],
            labels=["18-25", "26-40", "41-55", "56+"],
        ).astype("object")  # object pour laisser l'imputer catégoriel gérer les NaN

        # d) Feature d'interaction : combinaison categorie x region
        X["categorie_region"] = (
            X["categorie"].astype(str) + "_" + X["region"].astype(str)
        )

        # e) Remplacement des divisions par zéro / infinis générés en (a) et (b)
        X = X.replace([np.inf, -np.inf], np.nan)

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + [
            "revenu_par_annee_anciennete",
            "revenu_par_age",
            "tranche_age",
            "categorie_region",
        ]
