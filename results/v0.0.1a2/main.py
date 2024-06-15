import pandas as pd

import modelcomp as mc

import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold


def get_k_fold(seed=None, splits=5, repeats=5):
    return (
        RepeatedKFold(random_state=seed, n_splits=splits, n_repeats=repeats)
        if seed is not None
        else RepeatedKFold(n_splits=splits, n_repeats=repeats)
    )


def get_models(seed=None, seed_dict=None):
    return [
        RandomForestClassifier(random_state=(seed if seed is not None else None)),
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
    ]


if __name__ == "__main__":
    print("Program started")
    validation_count = 1
    # adding dictionary of model names abbreviations
    imported_abundance = pd.read_csv(
        os.getcwd() + "/../../data/abundance.csv", index_col=0
    )
    imported_meta = pd.read_csv(os.getcwd() + "/../../data/meta.csv", index_col=0)
    imported_abundance = mc.remove_falsy_columns(imported_abundance)
    imported_abundance = mc.remove_rare_features(imported_abundance)
    # reading data from csv
    positive_uniques = imported_meta["PatientGroup"].unique().tolist()
    positive_uniques.sort()
    positive_uniques.remove("0")
    # patient group possible results list creation
    X, y = mc.remove_string_columns(imported_abundance).to_numpy(), mc.encode(
        imported_meta.drop("SampleID", axis=1).to_numpy().ravel()
    )

    mc.cross_val_models(
        list(get_models(seed=42)),
        get_k_fold(seed=42),
        X,
        y,
        "multiclass",
        mc.remove_string_columns(imported_abundance).columns,
        explain=False,
    )
    print(f"Finished validation {validation_count} of 65")
    validation_count = validation_count + 1
    # multiclass
    abundance, meta = mc.filter_data(
        imported_abundance, imported_meta, "0", labels=positive_uniques
    )
    # setup list of labels in data
    abundance = mc.remove_string_columns(abundance)
    X, y = abundance.to_numpy(), meta.drop("SampleID", axis=1).to_numpy().ravel()
    mc.cross_val_models(
        list(get_models(seed=42)),
        get_k_fold(seed=42),
        X,
        y,
        positive_uniques,
        mc.remove_string_columns(imported_abundance).columns,
        explain=False,
    )
    # validation - train & test on all labels
    print(f"Finished validation {validation_count} of 65")
    validation_count += 1
    for train_index, train_label in enumerate(positive_uniques[1:]):
        abundance, meta = mc.filter_data(
            imported_abundance, imported_meta, "0", labels=[train_label]
        )
        # filtering labels by selected ones
        abundance = mc.remove_string_columns(abundance)
        X, y = abundance.to_numpy(), meta.drop("SampleID", axis=1).to_numpy().ravel()
        mc.cross_val_models(
            list(get_models(seed=42)),
            get_k_fold(seed=42),
            X,
            y,
            train_label,
            mc.remove_string_columns(imported_abundance).columns,
            explain=False,
        )
        # validation - train & test on the same label
        print(f"Finished validation {validation_count} of 65")
        validation_count = validation_count + 1
        for test_label in (
            positive_uniques[:train_index] + positive_uniques[train_index + 1 :]
        ):
            # iterating over positives without test label
            abundance, meta = mc.filter_data(
                imported_abundance,
                imported_meta,
                "0",
                label1=train_label,
                label2=test_label,
            )
            abundance, meta = abundance.drop("SampleID", axis=1), meta.drop(
                "SampleID", axis=1
            )
            control_train, control_test = mc.split_array(
                mc.get_label_indexes(meta, 0), seed=42
            )
            train, test = mc.get_label_indexes(meta, 1) + list(
                meta.loc[control_train].index.values
            ), mc.get_label_indexes(meta, 2) + list(meta.loc[control_test].index.values)
            train.sort()
            test.sort()
            X_train, X_test, y_train, y_test = (
                abundance.loc[train],
                abundance.loc[test],
                meta.loc[train],
                meta.loc[test],
            )
            y_test["PatientGroup"] = y_test["PatientGroup"].map(
                lambda x: 1 if x == 2 else x
            )
            mc.std_validation_models(
                list(get_models(seed=42)),
                X_train,
                X_test,
                y_train,
                y_test,
                train_label,
                test_label,
                mc.remove_string_columns(imported_abundance).columns,
            )
            print(f"Finished validation {validation_count} of 65")
            validation_count = validation_count + 1
    mc.general_plots(positive_uniques)
    print("Finished plotting general plots")
    print("Program finished")
