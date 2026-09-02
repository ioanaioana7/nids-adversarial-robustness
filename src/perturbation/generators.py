"""Generatoarele de variante perturbate.

Fiecare tip de perturbare respecta trei reguli, verificate ulterior de validators:
  1. modifica doar caracteristici pe care atacatorul le controleaza efectiv;
  2. are o DIRECTIE impusa (padding doar creste octetii, temporizarea doar
     creste durata, contoarele de conexiuni doar scad);
  3. propaga consistent caracteristicile derivate (vezi dependencies.propagate).

Nivelul 0 este intotdeauna identitatea, folosita ca control: predictiile pe el
trebuie sa reproduca exact baseline-ul O1.
"""

import numpy as np
import pandas as pd

from src import config
from src.perturbation import dependencies, schema


def _blank_ct_stats(n_rows: int) -> dict:
    """Statistici ct_state_ttl pentru tipurile care nu ating TTL-ul.

    Eticheta politicii e "none", nu "n/a": pandas interpreteaza "n/a" ca valoare
    lipsa la citirea CSV-ului, ceea ce ar face manifestul sa arate NaN.
    """
    return {"policy": "none", "resolved_exact": 0, "resolved_normal_fallback": 0,
            "resolved_constant_fallback": 0, "rows_held": int(n_rows)}


def identity(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Varianta de control: nicio modificare.

    Parameters:
        X (pd.DataFrame): fluxurile de atac eligibile.
        level (int): ignorat (identitatea nu are niveluri).
        ctx (dict): contextul cu lookup-urile ct_state_ttl.

    Returns:
        tuple[pd.DataFrame, dict]: o copie neatinsa a lui X si metadate.
    """
    return X.copy(), {"params": {}, "ct_state_ttl": _blank_ct_stats(len(X)), "propagation": {}}


def _perturb_ttl(X: pd.DataFrame, target_sttl: int, policy: str, ctx: dict,
                 target_dttl: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Nucleul comun al perturbarilor de TTL.

    Parameters:
        X (pd.DataFrame): fluxurile de plecare.
        target_sttl (int): valoarea sttl tinta.
        policy (str): "hold" sau "mimic", vezi dependencies.resolve_ct_state_ttl.
        ctx (dict): contextul cu lookup-urile.
        target_dttl (int | None): daca e setat, perturba si dttl (varianta
            deliberat nerealista, doar ca referinta de margine superioara).

    Returns:
        tuple[pd.DataFrame, dict]: varianta perturbata si metadate.
    """
    out = X.copy()
    out["sttl"] = np.int64(target_sttl)
    if target_dttl is not None:
        # dttl==0 inseamna "victima nu a raspuns"; nu inventam un raspuns inexistent.
        out["dttl"] = np.where(X["dttl"] > 0, np.int64(target_dttl), X["dttl"])

    ct_values, ct_stats = dependencies.resolve_ct_state_ttl(
        out, policy, ctx["ct_lookup"], ctx["normal_lookup"])
    out["ct_state_ttl"] = ct_values

    params = {"target_sttl": target_sttl, "ct_state_ttl_policy": policy}
    if target_dttl is not None:
        params["target_dttl"] = target_dttl
    return out, {"params": params, "ct_state_ttl": ct_stats, "propagation": {}}


def ttl_hold(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Perturbare TTL cu ct_state_ttl PASTRAT (margine conservatoare)."""
    return _perturb_ttl(X, config.TTL_LEVELS[level - 1], "hold", ctx)


def ttl_mimic(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Perturbare TTL cu ct_state_ttl adoptand semnatura tintei (margine optimista)."""
    return _perturb_ttl(X, config.TTL_LEVELS[level - 1], "mimic", ctx)


def ttl_both(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Referinta DELIBERAT NEREALISTA: perturba si dttl, care aparine victimei.

    Un atacator nu poate modifica TTL-ul pachetelor trimise de victima. Varianta
    exista doar ca margine superioara, pentru a cuantifica cat din dependenta
    modelului de TTL este pe ceva ce atacatorul nu poate atinge. Nu trebuie
    raportata ca rezultat de evaziune realizabil.
    """
    return _perturb_ttl(X, config.NORMAL_TTL_TARGET, "mimic", ctx,
                        target_dttl=config.NORMAL_DTTL_TARGET)


def padding(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Adauga umplutura benigna in payload-ul sursei: sbytes creste, niciodata nu scade.

    Daca umflarea pachetelor existente ar depasi MTU-ul, generatorul adauga
    pachete in loc sa le mareasca peste limita fizica, ceea ce modifica corect
    spkts si deci `rate`.
    """
    fraction = config.PADDING_LEVELS[level - 1]
    out = X.copy()
    new_sbytes = np.ceil(X["sbytes"].to_numpy(dtype=float) * (1.0 + fraction))

    # Plafon per pachet: nu coborim niciodata sub smean-ul original (o singura
    # inregistrare din setul de atac are deja smean=1504 > MTU).
    per_packet_cap = np.maximum(config.MTU_BYTES, X["smean"].to_numpy(dtype=float))
    min_packets = np.ceil(np.divide(new_sbytes, per_packet_cap,
                                    out=np.ones_like(new_sbytes),
                                    where=per_packet_cap > 0))
    new_spkts = np.maximum(X["spkts"].to_numpy(dtype=float), min_packets)

    out["sbytes"] = new_sbytes.astype(np.int64)
    out["spkts"] = new_spkts.astype(np.int64)
    out, prop_stats = dependencies.propagate(X, out)

    n_split = int((new_spkts > X["spkts"].to_numpy(dtype=float)).sum())
    return out, {
        "params": {"padding_fraction": fraction},
        "ct_state_ttl": _blank_ct_stats(len(X)),
        "propagation": {**prop_stats, "rows_needing_extra_packets": n_split},
    }


def timing(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Insereaza intarzieri intre pachete: dur creste, niciodata nu scade.

    Un atacator poate intotdeauna trimite mai lent; nu poate trimite mai repede
    decat permite reteaua. Fluxurile cu dur==0 nu pot fi dilatate (raportul ar fi
    nedefinit) si sunt lasate neatinse, numarate explicit in raport.
    """
    factor = config.TIMING_LEVELS[level - 1]
    out = X.copy()
    out["dur"] = X["dur"].to_numpy(dtype=float) * factor
    out, prop_stats = dependencies.propagate(X, out)
    return out, {
        "params": {"duration_factor": factor},
        "ct_state_ttl": _blank_ct_stats(len(X)),
        "propagation": prop_stats,
    }


def connection_rate(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Reduce contoarele de conexiuni: atacatorul incetineste scanarea/atacul.

    Contoarele ct_* sunt agregate pe o fereastra de 100 de conexiuni. Nu pot fi
    crescute arbitrar, dar pot fi intotdeauna reduse, incetinind atacul. Minimul
    observat in date este 1 (nu 0), deci acolo se opreste scaderea.
    """
    scale = config.CONNECTION_RATE_LEVELS[level - 1]
    out = X.copy()
    for col in config.CONNECTION_RATE_FEATURES:
        original = X[col].to_numpy(dtype=float)
        reduced = np.floor(original * scale) if scale > 0 else np.full_like(original, 0.0)
        clipped = np.maximum(reduced, config.MIN_CONNECTION_COUNT)
        # niciodata nu creste: pastreaza minimul dintre original si valoarea redusa
        out[col] = np.minimum(clipped, original).astype(np.int64)
    return out, {
        "params": {"connection_rate_scale": scale},
        "ct_state_ttl": _blank_ct_stats(len(X)),
        "propagation": {},
    }


def _combined(X: pd.DataFrame, level: int, ctx: dict, policy: str) -> tuple[pd.DataFrame, dict]:
    """TTL + padding + temporizare aplicate impreuna, pentru a testa supra-aditivitatea.

    TTL are 3 niveluri iar padding/temporizarea au 4; la nivelul 4 TTL-ul
    satureaza la cel mai agresiv nivel al sau (indexul e limitat superior).
    """
    ttl_index = min(level, len(config.TTL_LEVELS))
    out, ttl_meta = _perturb_ttl(X, config.TTL_LEVELS[ttl_index - 1], policy, ctx)

    fraction = config.PADDING_LEVELS[level - 1]
    factor = config.TIMING_LEVELS[level - 1]
    new_sbytes = np.ceil(X["sbytes"].to_numpy(dtype=float) * (1.0 + fraction))
    per_packet_cap = np.maximum(config.MTU_BYTES, X["smean"].to_numpy(dtype=float))
    min_packets = np.ceil(np.divide(new_sbytes, per_packet_cap,
                                    out=np.ones_like(new_sbytes),
                                    where=per_packet_cap > 0))
    new_spkts = np.maximum(X["spkts"].to_numpy(dtype=float), min_packets)
    out["sbytes"] = new_sbytes.astype(np.int64)
    out["spkts"] = new_spkts.astype(np.int64)
    out["dur"] = X["dur"].to_numpy(dtype=float) * factor

    out, prop_stats = dependencies.propagate(X, out)
    n_split = int((new_spkts > X["spkts"].to_numpy(dtype=float)).sum())
    return out, {
        "params": {**ttl_meta["params"], "padding_fraction": fraction,
                   "duration_factor": factor},
        "ct_state_ttl": ttl_meta["ct_state_ttl"],
        "propagation": {**prop_stats, "rows_needing_extra_packets": n_split},
    }


def combined_hold(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Combinatie de perturbari, cu ct_state_ttl pastrat (margine conservatoare)."""
    return _combined(X, level, ctx, "hold")


def combined_mimic(X: pd.DataFrame, level: int, ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Combinatie de perturbari, cu ct_state_ttl mimetizat (margine optimista)."""
    return _combined(X, level, ctx, "mimic")


# tip -> (functie, numar de niveluri, descriere scurta)
PERTURBATION_TYPES: dict[str, tuple] = {
    "identity": (identity, 1, "control: nicio modificare"),
    "ttl_hold": (ttl_hold, len(config.TTL_LEVELS), "sttl -> tinta, ct_state_ttl pastrat"),
    "ttl_mimic": (ttl_mimic, len(config.TTL_LEVELS), "sttl -> tinta, ct_state_ttl mimetizat"),
    "padding": (padding, len(config.PADDING_LEVELS), "umplutura in payload-ul sursei"),
    "timing": (timing, len(config.TIMING_LEVELS), "dilatarea duratei fluxului"),
    "connection_rate": (connection_rate, len(config.CONNECTION_RATE_LEVELS),
                        "reducerea contoarelor de conexiuni"),
    "combined_hold": (combined_hold, len(config.COMBINED_LEVELS),
                      "TTL+padding+timing, ct_state_ttl pastrat"),
    "combined_mimic": (combined_mimic, len(config.COMBINED_LEVELS),
                       "TTL+padding+timing, ct_state_ttl mimetizat"),
    "ttl_both": (ttl_both, 1, "NEREALIST: perturba si dttl (referinta margine sup.)"),
}


def build_context(*frames: pd.DataFrame) -> dict:
    """Pregateste lookup-urile necesare generatoarelor.

    O2/O3 apeleaza build_context(train, test): modelele sunt inghetate, deci
    statisticile senzorului nu intra niciodata in antrenare si folosirea ambelor
    partitii nu produce scurgere de date.

    O6 apeleaza build_context(train): acolo variantele perturbate DEVIN date de
    antrenare, deci orice statistica derivata din test ar fi scurgere. Cheia
    "source_frames" din context inregistreaza numarul de randuri per cadru
    sursa, ca apelantul sa poata verifica ulterior din ce a fost construit.

    Parameters:
        *frames (pd.DataFrame): cadrele brute din care se citesc maparile.

    Returns:
        dict: contextul cu maparile ct_state_ttl si provenienta lor.

    Raises:
        ValueError: daca nu se da niciun cadru.
    """
    if not frames:
        raise ValueError("build_context are nevoie de cel putin un cadru sursa")
    return {
        "ct_lookup": dependencies.build_ct_state_ttl_lookup(*frames),
        "normal_lookup": dependencies.build_normal_signature_lookup(*frames),
        "source_frames": [int(len(f)) for f in frames],
    }


def generate_variant(X: pd.DataFrame, perturbation_type: str, level: int,
                     ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Genereaza o singura varianta perturbata.

    API-ul pe care O3 il va folosi pentru a regenera variante in memorie, fara
    a depinde de matrici salvate pe disc (generarea e deterministica).

    Parameters:
        X (pd.DataFrame): fluxurile de atac eligibile (doar coloane de features).
        perturbation_type (str): cheie din PERTURBATION_TYPES.
        level (int): 0 = identitate; 1..n = nivelurile tipului respectiv.
        ctx (dict): contextul returnat de build_context().

    Returns:
        tuple[pd.DataFrame, dict]: varianta perturbata (aceeasi schema ca X) si metadate.

    Raises:
        ValueError: pentru un tip necunoscut sau un nivel in afara intervalului.
    """
    if perturbation_type not in PERTURBATION_TYPES:
        raise ValueError(f"tip de perturbare necunoscut: {perturbation_type!r}")
    func, n_levels, _ = PERTURBATION_TYPES[perturbation_type]

    if level == 0:
        return identity(X, 0, ctx)
    if not 1 <= level <= n_levels:
        raise ValueError(f"nivel {level} invalid pentru {perturbation_type!r} "
                         f"(disponibile: 1..{n_levels})")

    perturbed, meta = func(X, level, ctx)
    return schema.restore_dtypes(perturbed, X), meta


def variant_grid() -> list[tuple[str, int]]:
    """Grila completa de variante generate de O2.

    Returns:
        list[tuple[str, int]]: perechi (tip, nivel), incepand cu controlul identitate.
    """
    grid = [("identity", 0)]
    for name, (_, n_levels, _) in PERTURBATION_TYPES.items():
        if name == "identity":
            continue
        grid.extend((name, level) for level in range(1, n_levels + 1))
    return grid
