import pandas as pd

import modelcomp as mc

import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_k_fold(seed=None, splits=5, repeats=2):
    return (
        RepeatedKFold(random_state=seed, n_splits=splits, n_repeats=repeats)
        if seed is not None
        else RepeatedKFold(n_splits=splits, n_repeats=repeats)
    )


def get_models(seed=None, seed_dict=None):
    """
    xgb.XGBClassifier(random_state=(seed if seed is not None else None)),
    LogisticRegression(
        random_state=(seed if seed is not None else None), max_iter=10000
    ),
    SVC(
            random_state=(seed if seed is not None else None),
            kernel="linear",
            probability=True,
        ),
    KNeighborsClassifier(),
    """
    return [
        RandomForestClassifier(random_state=(seed if seed is not None else None)),
        DecisionTreeClassifier(random_state=(seed if seed is not None else None)),
    ]


if __name__ == "__main__":
    print("Program started")
    plt.rcParams["figure.constrained_layout.use"] = True
    validation_count = 1
    # adding dictionary of model names abbreviations
    imported_abundance = pd.read_csv(
        "/home/gil/projects/Alpha/modelcomp-appendix/data/abundance.csv", index_col=0
    )
    imported_meta = pd.read_csv(
        "/home/gil/projects/Alpha/modelcomp-appendix/data/meta.csv", index_col=0
    )
    imported_abundance = mc.utils.remove_falsy_columns(imported_abundance)
    imported_abundance = mc.utils.remove_rare_features(imported_abundance)
    # reading data from csv
    positive_uniques = imported_meta["PatientGroup"].unique().tolist()
    positive_uniques.sort()
    positive_uniques.remove("0")
    # patient group possible results list creation
    X, y = mc.utils.remove_string_columns(
        imported_abundance
    ).to_numpy(), mc.utils.encode(
        imported_meta.drop("SampleID", axis=1).to_numpy().ravel()
    )
    # multiclass
    abundance, meta = mc.utils.filter_data(
        imported_abundance, imported_meta, "0", labels=positive_uniques
    )
    # setup list of labels in data
    abundance = mc.utils.remove_string_columns(abundance)
    X, y = abundance.to_numpy(), meta.drop("SampleID", axis=1).to_numpy().ravel()
    X, y = StandardScaler().fit_transform(X), mc.utils.encode(y)
    mc_all = mc.storage.ModelComparison(
        list(get_models(seed=42)),
        get_k_fold(seed=42),
        X,
        y,
        mc.utils.remove_string_columns(imported_abundance).columns,
    )
    mc_all.validate(
        "accuracy_score",
        "roc_curve",
        "precision_recall_curve",
        "f1_score",
        "builtin_importance",
        "shap_explainer",
    )
    mc_all.export("export", "mean", "std", plots=True)
    mco = mc.storage.ModelComparison.from_path("export")
    mco.export("itmaybeworksidk", "mean", "std", plots=True)
