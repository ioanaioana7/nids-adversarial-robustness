"""
O2 — Generarea variantelor perturbate ale traficului malitios (UNSW-NB15)
========================================================================
Incarca modelele INGHETATE din O1 si genereaza variante perturbate ale
fluxurilor de atac din setul de test oficial. Nu reantreneaza nimic si nu
masoara evaziunea (aceea e O3): se opreste la "exista variante valide si
demonstrabil bine-formate".

Cum se ruleaza:
    python run_perturbation.py

Rezultate in results/perturbation/:
    perturbation_manifest.csv    parametrii si statisticile fiecarei variante
    validity_report.csv          verificarile de validitate, per varianta
    feature_deltas.csv           ce s-a schimbat efectiv si cu cat
    eligible_flags.csv           eligibilitatea per model, pentru O3
    ct_state_ttl_rule.json       maparea empirica ct_state_ttl + dovezi
    perturbation_summary.json    rezumatul rularii
    samples/<varianta>.csv       eșantion inspectabil per varianta
"""

from src import config, utils
from src.perturbation import runner


def main():
    logger = utils.setup_logging(log_path=config.PERTURBATION_LOG_PATH,
                                 logger_name="nids_perturbation")
    runner.run(logger)


if __name__ == "__main__":
    main()
