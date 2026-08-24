"""Verificarea validitatii variantelor perturbate.

Fiecare varianta trece prin toate verificarile de mai jos; o incalcare face
rularea sa cada zgomotos, pentru ca o varianta invalida ar produce o rata de
evaziune lipsita de sens in O3.
"""

import numpy as np
import pandas as pd

from src import config
from src.perturbation import schema


class ValidityError(AssertionError):
    """O varianta perturbata incalca o constrangere fizica sau de schema."""


def _rel_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Eroare relativa element-cu-element, ignorand pozitiile cu numitor ~0."""
    mask = np.abs(expected) > 1e-9
    err = np.zeros_like(expected, dtype=float)
    err[mask] = np.abs(actual[mask] - expected[mask]) / np.abs(expected[mask])
    return err


def check_directionality(original: pd.DataFrame, perturbed: pd.DataFrame) -> list[dict]:
    """Perturbarile au sens unic: octetii/durata doar cresc, contoarele doar scad."""
    findings = []
    for col in ["sbytes", "spkts", "dur"]:
        violations = int((perturbed[col].to_numpy() < original[col].to_numpy()).sum())
        findings.append({"check": f"monotonic_increase[{col}]", "violations": violations})
    for col in config.CONNECTION_RATE_FEATURES:
        violations = int((perturbed[col].to_numpy() > original[col].to_numpy()).sum())
        findings.append({"check": f"monotonic_decrease[{col}]", "violations": violations})
    return findings


def check_bounds(original: pd.DataFrame, perturbed: pd.DataFrame) -> list[dict]:
    """Limitele de domeniu: TTL livrabil, contoare >= 1, pachete sub MTU.

    Limitele se aplica RELATIV la datele originale: captura reala contine deja
    randuri in afara intervalelor "realiste" (61 fluxuri de atac au sttl in {0,1},
    o inregistrare are smean=1504 > MTU). Perturbarea nu are voie sa INRAUTATEASCA
    situatia, dar nu e penalizata pentru ce era deja acolo.
    """
    findings = []
    for col, (low, high) in schema.bounds().items():
        if col not in perturbed.columns:
            continue
        values = perturbed[col].to_numpy(dtype=float)
        base = original[col].to_numpy(dtype=float)
        violations = 0
        if low is not None:
            violations += int((values < np.minimum(low, base)).sum())
        if high is not None:
            violations += int((values > np.maximum(high, base)).sum())
        findings.append({"check": f"bounds[{col}]", "violations": violations})

    # smean nu poate depasi MTU-ul, dar nici nu inrautatim randurile care deja il depasesc
    cap = np.maximum(config.MTU_BYTES, original["smean"].to_numpy(dtype=float))
    over = int((perturbed["smean"].to_numpy(dtype=float) > cap).sum())
    findings.append({"check": "mtu[smean]", "violations": over})
    return findings


def check_immutability(original: pd.DataFrame, perturbed: pd.DataFrame,
                       extra_mutable: tuple[str, ...] = ()) -> list[dict]:
    """Coloanele victimei si cele care definesc identitatea atacului nu se schimba.

    Parameters:
        original (pd.DataFrame): cadrul de plecare.
        perturbed (pd.DataFrame): varianta generata.
        extra_mutable (tuple[str, ...]): coloane pe care ACEASTA varianta are voie
            sa le modifice in mod excepional (ex. dttl pentru referinta nerealista).
    """
    findings = []
    for col in schema.immutable_columns():
        if col in extra_mutable or col not in perturbed.columns:
            continue
        changed = int((perturbed[col].to_numpy() != original[col].to_numpy()).sum())
        findings.append({"check": f"immutable[{col}]", "violations": changed})
    return findings


def check_dependency_consistency(original: pd.DataFrame, perturbed: pd.DataFrame) -> list[dict]:
    """Identitatile algebraice se pastreaza la fel de bine ca in datele originale.

    Verificare INDEPENDENTA de mecanismul de propagare: propagarea foloseste
    rapoarte, iar aici se verifica formula directa (smean ~ sbytes/spkts,
    rate ~ (spkts+dpkts-1)/dur).

    Comparatia e PER RAND, nu pe maximul global. Datele brute contin randuri unde
    `rate` nu respecta deloc formula (2.8% din fluxurile de atac; ex. dur=7.15,
    spkts=2, dar rate=250000), iar o toleranta bazata pe maxim ar fi fost
    satisfacuta trivial de orice varianta. Cerinta corecta e ca propagarea sa nu
    inrauteasca NICIUN rand fata de cat de (in)consistent era deja.
    """
    findings = []
    usable = (original["dur"].to_numpy() > 0) & (original["spkts"].to_numpy() > 1) \
        & (perturbed["dur"].to_numpy() > 0) & (perturbed["spkts"].to_numpy() > 1)
    if usable.sum() == 0:
        return [{"check": "dependency[skipped_no_usable_rows]", "violations": 0}]

    o, p = original[usable], perturbed[usable]

    # smean e o coloana INTREAGA: rotunjirea introduce pana la 0.5 in valoare
    # absoluta, ceea ce pentru un smean mic (~28 octeti) ar depasi orice toleranta
    # relativa rezonabila. Se verifica deci absolut, rand cu rand.
    o_smean_exp = o["sbytes"].to_numpy(float) / o["spkts"].to_numpy(float)
    p_smean_exp = p["sbytes"].to_numpy(float) / p["spkts"].to_numpy(float)
    base_abs = np.abs(o["smean"].to_numpy(float) - o_smean_exp)
    var_abs = np.abs(p["smean"].to_numpy(float) - p_smean_exp)
    worsened = var_abs > base_abs + 1.0
    findings.append({"check": "dependency[smean~sbytes/spkts]",
                     "violations": int(worsened.sum()),
                     "rows_checked": int(len(base_abs)),
                     "max_abs_err_increase": float((var_abs - base_abs).max())})

    # rate e float. Propagarea prin rapoarte pastreaza eroarea relativa EXACT:
    #   rate' = rate * r_pktsum / r_dur  si  formula' = formula * r_pktsum / r_dur
    # deci rate'/formula' == rate/formula. Verificam tocmai asta, per rand.
    o_rate_exp = (o["spkts"].to_numpy(float) + o["dpkts"].to_numpy(float) - 1) / o["dur"].to_numpy(float)
    p_rate_exp = (p["spkts"].to_numpy(float) + p["dpkts"].to_numpy(float) - 1) / p["dur"].to_numpy(float)
    base_rel = _rel_error(o["rate"].to_numpy(float), o_rate_exp)
    var_rel = _rel_error(p["rate"].to_numpy(float), p_rate_exp)
    increase = var_rel - base_rel
    findings.append({"check": "dependency[rate~(spkts+dpkts-1)/dur]",
                     "violations": int((increase > config.PROPAGATION_RTOL).sum()),
                     "rows_checked": int(len(base_rel)),
                     "max_rel_err_increase": float(increase.max())})
    return findings


def check_finite(perturbed: pd.DataFrame) -> list[dict]:
    """Nicio valoare NaN sau infinita nu a fost introdusa de propagare."""
    numeric = perturbed.select_dtypes(include=[np.number])
    bad = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    return [{"check": "finite[all_numeric]", "violations": bad}]


def check_schema(perturbed: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    """Schema (coloane, ordine, dtype) e identica cu cea asteptata de modelul inghetat."""
    same_cols = list(perturbed.columns) == list(reference.columns)
    dtype_mismatch = int(sum(perturbed[c].dtype != reference[c].dtype for c in reference.columns))
    return [
        {"check": "schema[column_order]", "violations": 0 if same_cols else 1},
        {"check": "schema[dtypes]", "violations": dtype_mismatch},
    ]


def validate(original: pd.DataFrame, perturbed: pd.DataFrame, variant: str,
             extra_mutable: tuple[str, ...] = (), strict: bool = True) -> pd.DataFrame:
    """Ruleaza toate verificarile pe o varianta si raporteaza rezultatele.

    Parameters:
        original (pd.DataFrame): cadrul de plecare.
        perturbed (pd.DataFrame): varianta generata.
        variant (str): eticheta variantei, pentru raport.
        extra_mutable (tuple[str, ...]): coloane exceptate de la imutabilitate.
        strict (bool): daca True, ridica ValidityError la prima incalcare.

    Returns:
        pd.DataFrame: cate o linie per verificare, cu numarul de incalcari.

    Raises:
        ValidityError: daca strict=True si o verificare a esuat.
    """
    findings = (
        check_directionality(original, perturbed)
        + check_bounds(original, perturbed)
        + check_immutability(original, perturbed, extra_mutable)
        + check_dependency_consistency(original, perturbed)
        + check_finite(perturbed)
        + check_schema(perturbed, original)
    )
    report = pd.DataFrame(findings)
    report.insert(0, "variant", variant)

    failed = report[report["violations"] > 0]
    if strict and not failed.empty:
        detail = ", ".join(f"{r['check']}={r['violations']}" for _, r in failed.iterrows())
        raise ValidityError(f"[{variant}] verificari picate: {detail}")
    return report
