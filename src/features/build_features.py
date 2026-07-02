

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn custom qui crée de nouvelles colonnes à partir des
    colonnes brutes (déjà nettoyées structurellement, mais pouvant encore
    contenir des NaN -> ces NaN se propagent naturellement dans les nouvelles
    colonnes et seront traités par l'imputer du ColumnTransformer en aval).

    Pourquoi dans le pipeline et pas dans le nettoyage structurel ?
    - Ces features dérivées sont potentiellement le fruit d'une logique métier
      qu'on veut versionner avec le modèle.
    - Si un jour une feature dépend d'une statistique du train (ex: écart à la
      moyenne du groupe), il FAUT qu'elle soit ici pour être fit sur train
      uniquement. On adopte cette structure dès maintenant par cohérence,
      même si les features actuelles sont purement déterministes.
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
        # Utile si on veut inspecter les noms de colonnes en sortie du pipeline
        base = list(input_features) if input_features is not None else []
        return base + [
            "revenu_par_annee_anciennete",
            "revenu_par_age",
            "tranche_age",
            "categorie_region",
        ]
