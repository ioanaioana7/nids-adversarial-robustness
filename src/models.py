"""Constructia pipeline-urilor de model (Random Forest si XGBoost)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.preprocessing import RareCategoryGrouper, build_preprocessor


def build_rf_pipeline(numeric_cols) -> Pipeline:
    """Construieste pipeline-ul Random Forest: grupare categorii rare -> encoding -> clasificator.

    Parameters:
        numeric_cols (list[str]): numele coloanelor numerice.

    Returns:
        sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("rare", RareCategoryGrouper(columns=config.CATEGORICAL_COLUMNS,
                                      min_freq=config.MIN_CATEGORY_FREQUENCY)),
        ("prep", build_preprocessor(numeric_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            class_weight=config.RF_CLASS_WEIGHT,
            n_jobs=config.RF_N_JOBS,
            random_state=config.RANDOM_SEED)),
    ])


def build_xgb_pipeline(numeric_cols) -> Pipeline:
    """Construieste pipeline-ul XGBoost: grupare categorii rare -> encoding -> clasificator.

    Parameters:
        numeric_cols (list[str]): numele coloanelor numerice.

    Returns:
        sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("rare", RareCategoryGrouper(columns=config.CATEGORICAL_COLUMNS,
                                      min_freq=config.MIN_CATEGORY_FREQUENCY)),
        ("prep", build_preprocessor(numeric_cols)),
        ("clf", XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            tree_method=config.XGB_TREE_METHOD,
            n_jobs=config.XGB_N_JOBS,
            random_state=config.RANDOM_SEED)),
    ])
