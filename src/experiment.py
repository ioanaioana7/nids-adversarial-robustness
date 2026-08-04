"""Metadata de experiment: parametri, mapare etichete, nume caracteristici, rezumat dataset."""

from datetime import datetime, timezone

import pandas as pd

from src import config, utils


def save_experiment_metadata(numeric_cols: list, labels: list, rf_params: dict, xgb_params: dict) -> None:
    """Salveaza metadata completa a experimentului ca JSON (results/experiment_metadata.json).

    Parameters:
        numeric_cols (list[str]): coloanele numerice folosite ca features.
        labels (list): clasele attack_cat, in ordinea folosita la evaluare.
        rf_params (dict): hiperparametrii clasificatorului Random Forest antrenat.
        xgb_params (dict): hiperparametrii clasificatorului XGBoost antrenat.
    """
    metadata = {
        "dataset": "UNSW-NB15",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "model_parameters": {"random_forest": rf_params, "xgboost": xgb_params},
        "feature_list": {
            "categorical": config.CATEGORICAL_COLUMNS,
            "numeric": numeric_cols,
        },
        "class_labels": list(labels),
        "library_versions": utils.library_versions(),
    }
    utils.save_json(metadata, config.RESULTS_DIR / "experiment_metadata.json")


def save_dataset_summary(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list) -> None:
    """Salveaza un rezumat al dataset-ului ca JSON (results/dataset_summary.json).

    Parameters:
        train (pd.DataFrame): setul de antrenare (cu attack_cat prezent).
        test (pd.DataFrame): setul de test.
        numeric_cols (list[str]): coloanele numerice folosite ca features.
    """
    dist = train[config.TARGET].value_counts()
    present_ttl = [c for c in config.TTL_FEATURES if c in train.columns]
    summary = {
        "training_samples": int(len(train)),
        "testing_samples": int(len(test)),
        "num_numeric_features": len(numeric_cols),
        "num_categorical_features": len(config.CATEGORICAL_COLUMNS),
        "num_attack_classes": int(train[config.TARGET].nunique()),
        "class_imbalance_ratio": float(dist.max() / dist.min()),
        "ttl_features_available": present_ttl,
    }
    utils.save_json(summary, config.RESULTS_DIR / "dataset_summary.json")


def save_label_mapping(label_encoder) -> None:
    """Salveaza maparea eticheta -> cod numeric folosita de XGBoost, ca JSON.

    Parameters:
        label_encoder (sklearn.preprocessing.LabelEncoder): encoder-ul fitted pe y_train.
    """
    mapping = {cls: int(code) for code, cls in enumerate(label_encoder.classes_)}
    utils.save_json(mapping, config.RESULTS_DIR / "label_mapping.json")


def save_feature_names(prep) -> None:
    """Salveaza numele caracteristicilor rezultate dupa transformare (ordinea folosita de model).

    Previne erori de aliniere intre coloane la construirea experimentelor de perturbare (O2).

    Parameters:
        prep: ColumnTransformer-ul fitted (pentru get_feature_names_out()).
    """
    names = prep.get_feature_names_out()
    pd.Series(names, name="feature_name").to_csv(config.RESULTS_DIR / "feature_names.csv", index=False)
