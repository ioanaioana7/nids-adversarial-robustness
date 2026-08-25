"""
O3 — Masurarea ratei de evaziune (UNSW-NB15)
============================================
Aditiv fata de O1 (RF/XGBoost), O2 (perturbare) si Transformer: nu atinge si nu
reantreneaza niciun model. Incarca modelele inghetate prin calea determinista
din `src/inference.py`, regenereaza determinist variantele produse de O2 si
numara ce se intampla cu fiecare flux.

CE MASOARA
    Pentru fiecare (model, tip de perturbare, nivel), incadreaza fluxurile in
    cele trei rezultate din obiectivele lucrarii:
      (a) clasificat corect in continuare
      (b) devenit "Normal"  -> evaziune propriu-zisa
      (c) confundat cu alt tip de atac -> fragilitatea granitelor dintre clase

DENOMINATORUL
    Totul se raporteaza la fluxurile ELIGIBILE — cele pe care modelul le-a
    semnalat ca malitioase pe traficul CURAT. Un flux pe care modelul nu l-a
    prins niciodata nu poate evada; includerea lui ar transforma esecuri
    preexistente ale modelului in succese ale atacatorului.

INTERVALE, NU CIFRE UNICE
    Perturbarile care ating TTL-ul se raporteaza ca interval intre marginea
    conservatoare (`hold`) si cea optimista (`mimic`), pentru ca `ct_state_ttl`
    nu poate fi recalculat pentru un flux perturbat (vezi README).

Cum se ruleaza:
    python run_evasion.py

Rezultate in results/evasion/:
    evasion_summary.csv              rata si compozitia per (model, tip, nivel)
    evasion_headline.csv             tabel pivotat, doar variante realizabile
    evasion_bounds.csv               intervalele hold..mimic
    evasion_per_class.csv            defalcare per clasa de atac
    outcome_composition.csv          cele trei rezultate, in procente
    minimum_evasion_intensity.csv    cel mai mic nivel la care un flux evadeaza
    per_row/outcomes_<model>.csv.gz  rezultate per flux (necesare pentru O4)
    evasion_run_summary.json         rezumatul rularii
si figuri in results/figures/ (evasion_curves, evasion_outcomes_*, evasion_per_class_*).
"""

from src import config, utils
from src.evasion import runner


def main():
    logger = utils.setup_logging(log_path=config.EVASION_LOG_PATH,
                                 logger_name="nids_evasion")
    runner.run(logger)


if __name__ == "__main__":
    main()
