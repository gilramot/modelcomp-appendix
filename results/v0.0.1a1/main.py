import pandas as pd

import modelcomp as mc

import os

if __name__ == "__main__":
    print("Program started")
    seeds = {"RF": 42, "SVM": 42}
    validation_count = 1
    # adding dictionary of model names abbreviations
    imported_abundance = pd.read_csv(
        os.path.join(os.getcwd(), "..", "..", "data", "abundance.csv"), index_col=0
    )
    imported_meta = pd.read_csv(
        os.path.join(os.getcwd(), "..", "..", "data", "meta.csv"), index_col=0
    )
    imported_abundance = mc.remove_falsy_columns(imported_abundance)
    imported_abundance = mc.remove_rare_species(imported_abundance)
    # reading data from csv
    positive_uniques = imported_meta["PatientGroup"].unique().tolist()
    positive_uniques.sort()
    positive_uniques.remove("0")
    # patient group possible results list creation
    for model_name in mc.model_names:
        mc.make_dir(os.path.join("export", "A", "A", model_name, "data"))
        mc.make_dir(os.path.join("export", "A", "A", model_name, "plots"))
    for disease_name_test in positive_uniques:
        for disease_name_train in positive_uniques:
            for model_name in mc.model_names:
                mc.make_dir(
                    os.path.join(
                        "export",
                        disease_name_test,
                        disease_name_train,
                        model_name,
                        "data",
                    )
                )
                mc.make_dir(
                    os.path.join(
                        "export",
                        disease_name_test,
                        disease_name_train,
                        model_name,
                        "plots",
                    )
                )
    # making the directories (unneeded after 0.0.1a1)
    abundance, meta = mc.filter_data(
        imported_abundance, imported_meta, "0", disease_list=positive_uniques
    )
    # setup list of diseases in data
    abundance = mc.remove_string_columns(abundance)
    X, y = abundance.to_numpy(), meta.drop("SampleID", axis=1).to_numpy().ravel()
    #mc.cross_val_models(
    #    list(mc.get_models(seed_dict=seeds)),
    #    mc.get_k_fold(),
    #    X,
    #    y,
    #    positive_uniques,
    #    abundance.columns,
    #    explain=False
    #)
    # validation - train & test on all diseases
    print(f"Completed validation {validation_count} of 65")
    validation_count += 1
    for test_index, test_disease in enumerate(positive_uniques):
        abundance, meta = mc.filter_data(
            imported_abundance, imported_meta, "0", disease_list=[test_disease]
        )
        # filtering diseases by selected ones
        abundance = mc.remove_string_columns(abundance)
        X, y = abundance.to_numpy(), meta.drop("SampleID", axis=1).to_numpy().ravel()
        mc.cross_val_models(
            list(mc.get_models(seed_dict=seeds)),
            mc.get_k_fold(),
            X,
            y,
            test_disease,
            abundance.columns,
            explain=False
        )
        # validation - train & test on the same disease
        print(f"Completed validation {validation_count} of 65")
        validation_count = validation_count + 1
        for train_disease in (
            positive_uniques[:test_index] + positive_uniques[test_index + 1 :]
        ):
            # iterating over positives without test disease
            abundance, meta = mc.filter_data(
                imported_abundance,
                imported_meta,
                "0",
                disease1=train_disease,
                disease2=test_disease,
            )
            abundance, meta = abundance.drop("SampleID", axis=1), meta.drop(
                "SampleID", axis=1
            )
            control_train, control_test = mc.split_array(mc.get_label_indexes(meta, 0))
            train, test = mc.get_label_indexes(meta, 1) + list(
                meta.loc[control_train].index.values
            ), mc.get_label_indexes(meta, 2) + list(meta.loc[control_test].index.values)
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
                list(mc.get_models(seed_dict=seeds)),
                X_train,
                X_test,
                y_train,
                y_test,
                test_disease,
                train_disease,
                abundance.columns,
                explain=False
            )
            print(f"Completed validation {validation_count} of 65")
            validation_count = validation_count + 1
    mc.general_plots(positive_uniques)
    print("Completed plotting general plots")
    print("Program finished")
