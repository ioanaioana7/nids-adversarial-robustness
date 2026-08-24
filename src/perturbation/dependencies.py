"""Propagarea consistenta a caracteristicilor derivate si rezolvarea lui ct_state_ttl.

Propagare prin RAPOARTE, nu prin recalculare din formula. Motivul e empiric:
formulele Argus nu se reproduc cu constantele documentate (sload = sbytes*8/dur
greseste cu un factor 2 la mediana, dload cu ~14%), dar FORMA lor este
proportionala. Identitatea

    sload' = sload * (sbytes'/sbytes) * (dur/dur')

este exacta pentru orice sload = C * sbytes / dur, indiferent de constanta C
necunoscuta. Asta evita reverse-engineering-ul lui Argus si pastreaza fidelitatea
valorii stocate initial (verificat: chiar si `rate`, care se potriveste cu formula
la ~1e-8 relativ, difera in valoare absoluta din cauza stocarii pe float32).
"""

import numpy as np
import pandas as pd

from src import config

# Raportul folosit cand numitorul original e 0 (schimbare imposibil de propagat).
NEUTRAL_RATIO = 1.0


def _safe_ratio(new_values, old_values) -> np.ndarray:
    """Raport element-cu-element, cu 1.0 acolo unde numitorul e 0.

    Parameters:
        new_values (array-like): valorile noi (numarator).
        old_values (array-like): valorile originale (numitor).

    Returns:
        np.ndarray: raportul, cu NEUTRAL_RATIO unde numitorul e 0 sau nefinit.
    """
    new_arr = np.asarray(new_values, dtype=float)
    old_arr = np.asarray(old_values, dtype=float)
    ratio = np.divide(new_arr, old_arr, out=np.full_like(old_arr, NEUTRAL_RATIO),
                      where=old_arr != 0)
    return np.where(np.isfinite(ratio), ratio, NEUTRAL_RATIO)


def propagate(original: pd.DataFrame, perturbed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Recalculeaza caracteristicile derivate dupa o schimbare de sbytes/spkts/dur.

    Se aplica dupa ce generatorul a setat deja caracteristicile de baza in
    `perturbed`. Nu atinge ct_state_ttl (vezi resolve_ct_state_ttl).

    Parameters:
        original (pd.DataFrame): cadrul de plecare, neatins.
        perturbed (pd.DataFrame): cadrul cu caracteristicile de baza deja modificate.

    Returns:
        tuple[pd.DataFrame, dict]: cadrul cu derivatele propagate, plus statistici
            despre cazurile in care propagarea a fost neutralizata (numitor 0).
    """
    out = perturbed.copy()

    r_bytes = _safe_ratio(out.sbytes, original.sbytes)
    r_pkts = _safe_ratio(out.spkts, original.spkts)
    r_dur = _safe_ratio(out.dur, original.dur)
    # rate ~ (spkts + dpkts - 1) / dur ; dpkts nu se modifica niciodata
    r_pktsum = _safe_ratio(out.spkts + out.dpkts - 1, original.spkts + original.dpkts - 1)
    # sinpkt ~ dur / (spkts - 1) : numarul de intervale intre pachete
    r_gaps = _safe_ratio(out.spkts - 1, original.spkts - 1)

    out["smean"] = original.smean * r_bytes / r_pkts
    out["sload"] = original.sload * r_bytes / r_dur
    out["rate"] = original["rate"] * r_pktsum / r_dur
    out["sinpkt"] = original.sinpkt * r_dur / r_gaps
    out["sjit"] = original.sjit * r_dur

    # Antrenate cauzal de dilatarea duratei: daca atacatorul intarzie, raspunsurile
    # victimei se intind si ele in timp. dbytes/dpkts ramanand fixe, dmean nu se schimba.
    out["dload"] = original.dload / r_dur
    out["dinpkt"] = original.dinpkt * r_dur
    out["djit"] = original.djit * r_dur

    stats = {
        "rows_zero_sbytes": int((np.asarray(original.sbytes) == 0).sum()),
        "rows_zero_dur": int((np.asarray(original.dur) == 0).sum()),
        "rows_single_packet": int((np.asarray(original.spkts) <= 1).sum()),
        "rows_zero_packet_sum": int((np.asarray(original.spkts + original.dpkts - 1) == 0).sum()),
    }
    return out, stats


def build_ct_state_ttl_lookup(*frames: pd.DataFrame) -> pd.Series:
    """Construieste maparea (state, sttl, dttl) -> ct_state_ttl observata in date.

    Se foloseste train+test deliberat: modelam ce ar calcula SENZORUL pentru un
    flux dat, adica o proprietate a procesului de generare a datelor, nu a
    modelului. Nu se antreneaza nimic pe test, deci nu exista scurgere de date.

    Parameters:
        *frames (pd.DataFrame): cadrele din care se citeste maparea.

    Returns:
        pd.Series: indexata pe (state, sttl, dttl), valoarea = ct_state_ttl dominant.
    """
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(["state", "sttl", "dttl"])["ct_state_ttl"]
    return grouped.agg(lambda s: s.value_counts().idxmax())


def build_normal_signature_lookup(*frames: pd.DataFrame) -> pd.Series:
    """Construieste maparea sttl -> ct_state_ttl tipic pentru TRAFIC NORMAL.

    Folosita ca rezerva pentru politica "mimic": daca (state, sttl, dttl) nu a
    fost niciodata observat, presupunem ca atacatorul care adopta acel sttl
    capata semnatura pe care o are traficul normal la acelasi sttl.

    Parameters:
        *frames (pd.DataFrame): cadrele din care se citeste maparea.

    Returns:
        pd.Series: indexata pe sttl, valoarea = ct_state_ttl dominant la trafic normal.
    """
    combined = pd.concat(frames, ignore_index=True)
    normal = combined[combined[config.TARGET] == "Normal"]
    return normal.groupby("sttl")["ct_state_ttl"].agg(lambda s: s.value_counts().idxmax())


def resolve_ct_state_ttl(perturbed: pd.DataFrame, policy: str,
                         lookup: pd.Series, normal_lookup: pd.Series) -> tuple[pd.Series, dict]:
    """Determina ct_state_ttl pentru un cadru in care sttl/dttl s-au schimbat.

    ct_state_ttl NU e o functie de intervale TTL (sttl=62 -> 2, dar sttl=63 -> 0;
    sttl=254 -> 2, dar sttl=252 -> 0), ci un contor pe fereastra glisanta de 100
    de conexiuni, imposibil de recalculat din inregistrari de flux izolate. De
    aceea se emit ambele margini, ca variante etichetate separat.

    Parameters:
        perturbed (pd.DataFrame): cadrul cu sttl/dttl deja modificate.
        policy (str): "hold" (pastreaza valoarea originala) sau "mimic"
            (adopta semnatura tintei).
        lookup (pd.Series): maparea (state, sttl, dttl) -> ct_state_ttl.
        normal_lookup (pd.Series): maparea sttl -> ct_state_ttl la trafic normal.

    Returns:
        tuple[pd.Series, dict]: valorile ct_state_ttl si statistici despre
            calea de rezolvare folosita (exact / normal / fallback).

    Raises:
        ValueError: pentru o politica necunoscuta.
    """
    if policy not in config.CT_STATE_TTL_POLICIES:
        raise ValueError(f"politica ct_state_ttl necunoscuta: {policy!r}")

    if policy == "hold":
        return perturbed["ct_state_ttl"], {
            "policy": policy, "resolved_exact": 0,
            "resolved_normal_fallback": 0, "resolved_constant_fallback": 0,
            "rows_held": int(len(perturbed)),
        }

    keys = pd.MultiIndex.from_arrays(
        [perturbed["state"], perturbed["sttl"], perturbed["dttl"]])
    exact = lookup.reindex(keys)
    exact_hit = exact.notna().to_numpy()

    from_normal = normal_lookup.reindex(perturbed["sttl"]).to_numpy(dtype=float)
    normal_hit = ~np.isnan(from_normal) & ~exact_hit

    values = np.where(exact_hit, exact.to_numpy(dtype=float),
                      np.where(normal_hit, from_normal, config.CT_STATE_TTL_FALLBACK))

    stats = {
        "policy": policy,
        "resolved_exact": int(exact_hit.sum()),
        "resolved_normal_fallback": int(normal_hit.sum()),
        "resolved_constant_fallback": int(len(perturbed) - exact_hit.sum() - normal_hit.sum()),
        "rows_held": 0,
    }
    return pd.Series(values, index=perturbed.index), stats
