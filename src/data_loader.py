"""Incarcarea datelor brute UNSW-NB15 (partitia train/test predefinita)."""

import pandas as pd

from src import config


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Incarca seturile de train/test din config.TRAIN_PATH / config.TEST_PATH.

    id, attack_cat si label raman in cadrele returnate (necesare pentru EDA
    si pentru exportul de predictii); sunt excluse din matricea de features
    abia in main(), la construirea lui X_train/X_test.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train, test)
    """
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    for col in config.DROP_COLUMNS:
        train = train.drop(columns=col, errors="ignore")
        test = test.drop(columns=col, errors="ignore")
    return train, test
