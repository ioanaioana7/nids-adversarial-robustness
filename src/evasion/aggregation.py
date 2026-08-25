"""Tabelele agregate: cifre principale, intervale hold/mimic, intensitate minima de evaziune."""

import numpy as np
import pandas as pd

from src import config


def headline_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Rata de evaziune per (model, tip, nivel), doar pentru variantele realizabile.

    Exclude ttl_both, care perturba si dttl — TTL-ul victimei, inaccesibil
    atacatorului. Ramane in fisierul complet, etichetat, ca referinta.

    Parameters:
        summary (pd.DataFrame): rezultatul lui run_sweep.

    Returns:
        pd.DataFrame: tabel pivotat, tipuri pe randuri, niveluri pe coloane.
    """
    realizable = summary[summary["realizable"] & (summary["level"] > 0)]
    return realizable.pivot_table(index=["model", "type"], columns="level",
                                  values="evasion_rate").round(3)


def bounds_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Intervalele hold/mimic pentru perturbarile care ating ct_state_ttl.

    ct_state_ttl nu poate fi recalculat pentru un flux perturbat (vezi README),
    asa ca O2 a emis ambele margini. Rezultatul corect de raportat este
    INTERVALUL, nu una dintre margini: distanta dintre ele masoara cat din
    robustetea modelului depinde de o caracteristica pe care nici atacatorul
    nu o controleaza direct, nici aparatorul nu o poate reconstitui.

    Returns:
        pd.DataFrame: o linie per (model, familie, nivel), cu marginea inferioara
            (hold, conservatoare) si cea superioara (mimic, optimista).
    """
    rows = []
    for family, (lower_type, upper_type) in config.EVASION_BOUND_PAIRS.items():
        lower = summary[summary["type"] == lower_type]
        upper = summary[summary["type"] == upper_type]
        for _, low in lower[lower["level"] > 0].iterrows():
            match = upper[(upper["model"] == low["model"]) & (upper["level"] == low["level"])]
            if match.empty:
                continue
            high = match.iloc[0]
            rows.append({
                "model": low["model"], "family": family, "level": int(low["level"]),
                "evasion_lower_hold": low["evasion_rate"],
                "evasion_upper_mimic": high["evasion_rate"],
                "interval_width": round(high["evasion_rate"] - low["evasion_rate"], 4),
            })
    return pd.DataFrame(rows).sort_values(["model", "family", "level"]).reset_index(drop=True)


def minimum_evasion_intensity(per_row: pd.DataFrame) -> pd.DataFrame:
    """Cel mai mic nivel la care fiecare flux evadeaza, per tip de perturbare.

    Raspunde la "cat de multa modificare e necesara", nu doar "se poate evada".
    Fluxurile care nu evadeaza la niciun nivel primesc NaN si sunt numarate separat.

    Parameters:
        per_row (pd.DataFrame): rezultatele per rand din masuratoare.

    Returns:
        pd.DataFrame: rezumat per (model, tip): cate fluxuri evadeaza si la ce nivel median.
    """
    evaded = per_row[(per_row["outcome"] == config.OUTCOME_EVADED) & (per_row["level"] > 0)]
    first = (evaded.groupby(["model", "type", "sample_id"])["level"].min()
             .rename("first_evasion_level").reset_index())

    totals = (per_row[per_row["level"] > 0]
              .groupby(["model", "type"])["sample_id"].nunique().rename("n_eligible"))

    rows = []
    for (model, ptype), group in first.groupby(["model", "type"]):
        total = int(totals.loc[(model, ptype)])
        rows.append({
            "model": model, "type": ptype,
            "n_eligible": total,
            "n_ever_evaded": len(group),
            "pct_ever_evaded": round(100.0 * len(group) / total, 3) if total else 0.0,
            "median_first_level": float(group["first_evasion_level"].median()),
            "n_evaded_at_level_1": int((group["first_evasion_level"] == 1).sum()),
        })
    return pd.DataFrame(rows).sort_values(["model", "type"]).reset_index(drop=True)


def outcome_composition(summary: pd.DataFrame) -> pd.DataFrame:
    """Compozitia celor trei rezultate per (model, tip, nivel), pentru graficele stivuite."""
    columns = ["model", "type", "level", "variant", "n_eligible", "realizable"]
    columns += [f"pct_{name}" for name in config.OUTCOME_ORDER]
    return summary[columns].sort_values(["model", "type", "level"]).reset_index(drop=True)


def sanity_check_identity(summary: pd.DataFrame, logger) -> dict:
    """Verifica invariantul: la nivelul 0 (identitate) rata de evaziune trebuie sa fie exact 0.

    Fluxurile eligibile sunt, prin definitie, cele pe care modelul le-a semnalat
    ca fiind malitioase pe traficul curat. Varianta-identitate nu modifica nimic,
    deci niciunul dintre ele nu are cum sa devina "Normal". O valoare nenula ar
    insemna ca lantul masuratorii nu se aliniaza cu artefactele de referinta.

    Raises:
        AssertionError: daca vreun model raporteaza evaziune la nivelul 0.
    """
    identity = summary[summary["level"] == 0]
    failures = identity[identity["evasion_rate"] > 0]
    if not failures.empty:
        detail = ", ".join(f"{r['model']}={r['evasion_rate']}%" for _, r in failures.iterrows())
        raise AssertionError(
            f"varianta-identitate raporteaza evaziune nenula ({detail}); "
            f"masuratoarea nu e aliniata cu predictiile de referinta")

    result = {row["model"]: {"evasion_rate": row["evasion_rate"],
                            "n_eligible": int(row["n_eligible"])}
              for _, row in identity.iterrows()}
    for model, values in result.items():
        logger.info(f"  [identitate] {model}: {values['n_eligible']:,} fluxuri eligibile, "
                    f"evaziune {values['evasion_rate']:.2f}%")
    return result
