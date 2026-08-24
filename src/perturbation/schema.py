"""Taxonomia caracteristicilor si contractul de schema pentru variantele perturbate.

Clasifica cele 42 de coloane de intrare dupa cine le controleaza in realitate
(atacator / victima / derivate / identitate), astfel incat generatoarele sa nu
poata produce fluxuri fizic imposibile.
"""

import pandas as pd

from src import config

# Coloanele excluse din matricea de features (vezi main.py).
NON_FEATURE_COLUMNS = [config.ID_COL, config.TARGET, config.LABEL_COL]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Coloanele de features, in ordinea in care modelul inghetat le primeste.

    Parameters:
        frame (pd.DataFrame): un cadru brut incarcat cu data_loader.load_data().

    Returns:
        list[str]: numele coloanelor de features (fara id/attack_cat/label).
    """
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def mutable_columns() -> list[str]:
    """Coloanele pe care O2 are voie sa le atinga (direct sau prin propagare)."""
    return sorted(set(
        config.SOURCE_CONTROLLED
        + config.CONNECTION_RATE_FEATURES
        + config.DERIVED_FEATURES
        + config.CAUSALLY_FORCED_BY_DUR
    ))


def immutable_columns() -> list[str]:
    """Coloanele care trebuie sa ramana identice bit-cu-bit in orice varianta."""
    return sorted(set(config.VICTIM_CONTROLLED + config.IDENTITY_DEFINING))


def integer_columns(frame: pd.DataFrame) -> list[str]:
    """Coloanele de features cu dtype intreg, care necesita rotunjire dupa propagare.

    Modelul inghetat a fost antrenat cu smean/spkts/sbytes/sttl/ct_* ca int64;
    propagarea prin rapoarte produce float, deci trebuie readuse la intreg.

    Parameters:
        frame (pd.DataFrame): cadrul de referinta din care se citesc dtype-urile.

    Returns:
        list[str]: numele coloanelor intregi.
    """
    cols = feature_columns(frame)
    return [c for c in cols if pd.api.types.is_integer_dtype(frame[c])]


def bounds() -> dict[str, tuple]:
    """Limitele de domeniu impuse per coloana, ca (minim, maxim); None = nelimitat.

    Returns:
        dict[str, tuple]: coloana -> (min, max).
    """
    limits: dict[str, tuple] = {
        "sttl": (config.MIN_REALISTIC_TTL, config.MAX_TTL),
        "dttl": (0, config.MAX_TTL),
        "spkts": (1, None),
        "sbytes": (0, None),
        "dur": (0.0, None),
        "smean": (0, None),
        "sload": (0.0, None),
        "rate": (0.0, None),
        "sinpkt": (0.0, None),
        "sjit": (0.0, None),
    }
    for col in config.CONNECTION_RATE_FEATURES:
        limits[col] = (config.MIN_CONNECTION_COUNT, None)
    return limits


def restore_dtypes(perturbed: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Readuce dtype-urile si ordinea coloanelor la cele asteptate de modelul inghetat.

    Parameters:
        perturbed (pd.DataFrame): cadrul perturbat (poate avea coloane promovate la float).
        reference (pd.DataFrame): cadrul original, sursa de adevar pentru dtype/ordine.

    Returns:
        pd.DataFrame: cadrul perturbat, cu aceeasi schema ca referinta.
    """
    out = perturbed.reindex(columns=reference.columns)
    for col in reference.columns:
        if pd.api.types.is_integer_dtype(reference[col]) and not pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].round().astype(reference[col].dtype)
        elif out[col].dtype != reference[col].dtype:
            out[col] = out[col].astype(reference[col].dtype)
    return out
