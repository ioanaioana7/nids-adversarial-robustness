"""Perturbare cu parametri arbitrari, pentru testarea interactiva din O5.

DE CE EXISTA SI CE **NU** FACE
    O2 genereaza o grila fixa de variante (niveluri predefinite). Interfata are
    nevoie de valori arbitrare: orice sttl intre 16 si 255, orice procent de
    padding, orice factor de temporizare.

    Modulul NU reimplementeaza logica de perturbare. Compune exact aceleasi
    primitive publice validate in O2:
        dependencies.propagate            — propagarea prin rapoarte
        dependencies.resolve_ct_state_ttl — politicile hold / mimic
        schema.restore_dtypes             — schema asteptata de modele
        validators.validate               — verificarile de validitate fizica

    Consecinta importanta: o perturbare construita aici trece prin ACELEASI
    verificari ca variantele masurate in O3. Daca ar incalca o constrangere
    (padding care scade octetii, coloana victimei modificata, derivate
    inconsistente), validatorul o respinge — deci interfata nu poate arata o
    "evaziune" obtinuta cu un flux imposibil fizic.
"""

import numpy as np
import pandas as pd

from src import config
from src.perturbation import dependencies, schema, validators


def apply_custom_perturbation(X: pd.DataFrame, ctx: dict, *,
                              target_sttl: int | None = None,
                              ct_policy: str = "mimic",
                              padding_fraction: float = 0.0,
                              duration_factor: float = 1.0,
                              connection_rate_scale: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Aplica o combinatie arbitrara de perturbari, respectand constrangerile O2.

    Parameters:
        X (pd.DataFrame): fluxurile de plecare (doar coloane de features).
        ctx (dict): contextul cu lookup-urile ct_state_ttl (generators.build_context).
        target_sttl (int | None): valoarea sttl tinta; None = nemodificat.
        ct_policy (str): "hold" sau "mimic", relevant doar daca sttl se schimba.
        padding_fraction (float): fractia adaugata la sbytes (0 = fara padding).
        duration_factor (float): factorul de dilatare a duratei (1 = neschimbat).
        connection_rate_scale (float): factorul de reducere a contoarelor ct_* (1 = neschimbat).

    Returns:
        tuple[pd.DataFrame, dict]: varianta perturbata si metadatele aplicarii.
    """
    out = X.copy()
    applied = {}

    if target_sttl is not None and int(target_sttl) != -1:
        out["sttl"] = np.int64(int(target_sttl))
        values, ct_stats = dependencies.resolve_ct_state_ttl(
            out, ct_policy, ctx["ct_lookup"], ctx["normal_lookup"])
        out["ct_state_ttl"] = values
        applied["target_sttl"] = int(target_sttl)
        applied["ct_policy"] = ct_policy
        applied["ct_resolution"] = ct_stats

    if padding_fraction > 0:
        new_sbytes = np.ceil(X["sbytes"].to_numpy(dtype=float) * (1.0 + padding_fraction))
        # Acelasi plafon MTU ca in O2: daca umflarea pachetelor existente ar depasi
        # limita fizica, se adauga pachete in loc sa se treaca peste ea.
        per_packet_cap = np.maximum(config.MTU_BYTES, X["smean"].to_numpy(dtype=float))
        min_packets = np.ceil(np.divide(new_sbytes, per_packet_cap,
                                        out=np.ones_like(new_sbytes),
                                        where=per_packet_cap > 0))
        new_spkts = np.maximum(X["spkts"].to_numpy(dtype=float), min_packets)
        out["sbytes"] = new_sbytes.astype(np.int64)
        out["spkts"] = new_spkts.astype(np.int64)
        applied["padding_fraction"] = float(padding_fraction)
        applied["rows_needing_extra_packets"] = int(
            (new_spkts > X["spkts"].to_numpy(dtype=float)).sum())

    if duration_factor > 1.0:
        out["dur"] = X["dur"].to_numpy(dtype=float) * duration_factor
        applied["duration_factor"] = float(duration_factor)

    if connection_rate_scale < 1.0:
        for column in config.CONNECTION_RATE_FEATURES:
            original = X[column].to_numpy(dtype=float)
            reduced = np.floor(original * connection_rate_scale)
            clipped = np.maximum(reduced, config.MIN_CONNECTION_COUNT)
            out[column] = np.minimum(clipped, original).astype(np.int64)
        applied["connection_rate_scale"] = float(connection_rate_scale)

    # Propagarea derivatelor e obligatorie oricand s-au schimbat sbytes/spkts/dur.
    if padding_fraction > 0 or duration_factor > 1.0:
        out, propagation = dependencies.propagate(X, out)
        applied["propagation"] = propagation

    return schema.restore_dtypes(out, X), applied


def check_validity(X: pd.DataFrame, perturbed: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """Trece varianta prin verificarile de validitate fizica din O2.

    Parameters:
        X (pd.DataFrame): fluxurile originale.
        perturbed (pd.DataFrame): varianta construita interactiv.

    Returns:
        tuple[bool, pd.DataFrame]: (totul e valid, raportul detaliat).
    """
    report = validators.validate(X, perturbed, "interactive", strict=False)
    return bool((report["violations"] == 0).all()), report


def classify(true_labels, predictions) -> np.ndarray:
    """Incadreaza in cele trei rezultate, folosind taxonomia din O3."""
    from src.evasion.outcomes import classify_outcomes
    return classify_outcomes(true_labels, predictions)
