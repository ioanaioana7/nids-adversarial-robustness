"""
O6 — Antrenare adversariala prin augmentare (UNSW-NB15)
=======================================================
Contrapartida defensiva a lui O3. Intrebarea: daca reantrenam pe trafic de atac
perturbat, se inchide diferenta de evaziune si cu ce cost pe traficul curat?

ADITIV. Nu atinge si nu suprascrie niciun artefact O1-O5. Modelele noi merg in
models/adversarial/, rezultatele in results/adversarial/.

TREI BRATE, PENTRU CA REZULTATUL SA FIE APARABIL
    O1   date originale                                   — referinta
    C1   original + duplicate de atac NEPERTURBATE        — control
    O6   original + copii de atac PERTURBATE              — aparerea
C1 adauga exact acelasi numar de randuri ca O6, deci un castig care apare si la
C1 se datoreaza volumului de date, nu perturbarii. Cifra care conteaza este
diferenta O6 − C1.

Un al patrulea brat, O6-LOO, e antrenat fara nicio varianta din familia TTL si
masurat tocmai pe ea: robustetea pe un tip NEVAZUT la antrenare arata ca modelul
a invatat invarianta, nu transformarile concrete.

CE NU ESTE. Nu e antrenare pe gradient (FGSM/PGD): gradientul cere un acces pe
care atacatorul din modelul de amenintare nu il are si produce fluxuri fizic
nerealizabile. Nu acopera atacatori adaptivi, care s-ar reoptimiza impotriva
modelului aparat — e o singura runda a jocului atac/aparare, nu echilibrul lui.

UTILIZARE
    python train_adversarial.py                  # toate etapele, in ordine
    python train_adversarial.py --stage prepare  # doar datele + manifestul
    python train_adversarial.py --stage trees
    python train_adversarial.py --stage transformer
    python train_adversarial.py --stage evaluate
    python train_adversarial.py --max-epochs 2   # proba scurta

RELUARE. Fiecare model deja salvat e sarit, la fel fiecare rulare deja evaluata,
deci o intrerupere costa cel mult bratul in curs.

REZULTATE in results/adversarial/:
    o6_manifest.json                 configuratia + criteriile pre-inregistrate
    clean_metrics.csv                performanta pe test curat, per rulare
    clean_per_class_f1.json          F1 per clasa, per rulare
    cohort_sizes.json                marimea cohortei comune, global si per clasa
    cohort_evasion_per_seed.csv      evaziune pe cohorta, per (brat, seed, varianta)
    cohort_evasion_aggregated.csv    aceleasi, mediate peste seed-uri
    own_eligible_evasion.csv         ratele brute, fiecare pe eligibilii proprii
    worst_case_*.csv                 cel mai rau caz realizabil, per (model, brat)
    o6_verdict.json                  confruntarea cu criteriile pre-inregistrate
    outcomes/                        matricile per rand, pentru re-analiza
si figuri in results/figures/ (o6_evasion_curves_*, o6_cost_vs_robustness).
"""

import argparse
import json
import os

# Trebuie setat INAINTE de orice import care initializeaza cuBLAS, altfel
# torch.use_deterministic_algorithms(True) esueaza pe operatiile de matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd                                          # noqa: E402

from src import config, utils                                # noqa: E402
from src.adversarial import report, runner                   # noqa: E402

STAGES = ["prepare", "trees", "transformer", "evaluate"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="O6 — antrenare adversariala prin augmentare, cu brat de control.")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all",
                        help="ruleaza o singura etapa (implicit: toate)")
    parser.add_argument("--max-epochs", type=int, default=config.TRANSFORMER_MAX_EPOCHS,
                        help=f"plafon de epoci pentru retea (implicit "
                             f"{config.TRANSFORMER_MAX_EPOCHS})")
    return parser.parse_args()


def finalize(state: dict, evaluated: dict, cohorts: dict, logger) -> dict:
    """Tabelele finale, figurile si verdictul fata de criteriile pre-inregistrate."""
    per_seed = cohorts["per_seed"]
    clean = report.clean_summary(evaluated["clean"])
    per_class = report.per_class_f1_table(
        json.loads((config.ADVERSARIAL_DIR / "clean_per_class_f1.json").read_text("utf-8")))

    worst_global = report.worst_case_table(per_seed, "evasion_rate")
    worst_class = report.worst_case_table(per_seed, f"evasion_{config.O6_CRITERION_CLASS}")

    clean.to_csv(config.ADVERSARIAL_DIR / "clean_summary.csv", index=False)
    per_class.to_csv(config.ADVERSARIAL_DIR / "clean_per_class_summary.csv", index=False)
    worst_global.to_csv(config.ADVERSARIAL_DIR / "worst_case_global.csv", index=False)
    worst_class.to_csv(config.ADVERSARIAL_DIR / "worst_case_criterion_class.csv", index=False)

    holdout = report.holdout_comparison(per_seed)
    if not holdout.empty:
        holdout.to_csv(config.ADVERSARIAL_DIR / "holdout_generalization.csv", index=False)

    figures = report.save_figures(per_seed, clean, config.FIGURES_DIR)
    logger.info(f"\n[salvat] figuri: {', '.join(figures)}")

    verdicts = {}
    for model in sorted(clean["model"].unique()):
        verdicts[model] = report.verdict(worst_global, worst_class, clean, per_class,
                                         cohorts["cohort_sizes"], model=model)
    payload = {
        "preregistered_in": str(config.O6_MANIFEST_PATH.name),
        "primary_model": "Transformer",
        "verdicts": verdicts,
        "holdout_generalization": holdout.to_dict("records") if not holdout.empty else [],
    }
    utils.save_json(payload, config.ADVERSARIAL_DIR / "o6_verdict.json")

    log_verdict(clean, worst_global, worst_class, verdicts, cohorts, logger)
    return payload


def log_verdict(clean, worst_global, worst_class, verdicts, cohorts, logger) -> None:
    """Rezumatul citibil in consola si in adversarial.log."""
    logger.info("\n" + "=" * 74)
    logger.info("REZULTATE — trafic curat (medie peste seed-uri)")
    logger.info("=" * 74)
    logger.info(f"  {'model':<14}{'brat':<9}{'macro F1':>10}{'±':>9}{'weighted F1':>13}")
    for _, row in clean.sort_values(["model", "arm"]).iterrows():
        logger.info(f"  {row['model']:<14}{row['arm']:<9}{row['macro_f1_mean']:>10.4f}"
                    f"{row['macro_f1_std']:>9.4f}{row['weighted_f1_mean']:>13.4f}")

    logger.info("\n" + "=" * 74)
    logger.info("REZULTATE — cel mai rau caz de evaziune, cohorta comuna")
    logger.info("=" * 74)
    for model, sizes in cohorts["cohort_sizes"].items():
        logger.info(f"  {model}: cohorta n={sizes['n']:,} "
                    f"({sizes['pct_of_attacks']:.2f}% din fluxurile de atac)")
    logger.info(f"\n  {'model':<14}{'brat':<9}{'global %':>10}"
                f"{config.O6_CRITERION_CLASS + ' %':>12}{'varianta':>18}")
    for _, row in worst_global.sort_values(["model", "arm"]).iterrows():
        match = worst_class[(worst_class["model"] == row["model"])
                            & (worst_class["arm"] == row["arm"])]
        class_value = float(match.iloc[0]["worst_case_mean"]) if not match.empty else float("nan")
        logger.info(f"  {row['model']:<14}{row['arm']:<9}{row['worst_case_mean']:>10.2f}"
                    f"{class_value:>12.2f}{row['worst_variants'][:18]:>18}")

    logger.info("\n" + "=" * 74)
    logger.info("VERDICT fata de criteriile PRE-INREGISTRATE")
    logger.info("=" * 74)
    for model, v in verdicts.items():
        primary = v["primary_robustness"]
        cost = v["primary_clean_cost"]
        safety = v["safety"]
        mark = lambda ok: "TRECUT" if ok else "PICAT"  # noqa: E731
        logger.info(f"\n  {model} (cohorta n={v['cohort_n']:,})")
        logger.info(f"    robustete : C1={primary['c1_pct']:.2f}% -> O6={primary['o6_pct']:.2f}%  "
                    f"(castig {primary['improvement_over_c1_pp']:+.2f} pp, prag "
                    f"{primary['threshold_pp']:.0f}) ... {mark(primary['passed'])}")
        logger.info(f"    cost curat: macro-F1 O6={cost['o6_macro_f1']:.4f}, scadere fata de "
                    f"C1 {cost['drop_vs_c1']:+.4f} / O1 {cost['drop_vs_o1']:+.4f} "
                    f"(prag {cost['threshold']}) ... {mark(cost['passed'])}")
        logger.info(f"    siguranta : cea mai mare pierdere per clasa "
                    f"{safety['worst_drop']:+.4f} pe {safety['worst_class']} "
                    f"(prag {safety['threshold']}) ... {mark(safety['passed'])}")
        logger.info(f"    tinta aspirationala (<= {config.O6_STRETCH_EVASION_ABS}%): "
                    f"{v['stretch_target']['value_pct']:.2f}% ... "
                    f"{'atinsa' if v['stretch_target']['met'] else 'neatinsa'}")
    logger.info("\n" + "=" * 74)


def main():
    args = parse_args()
    utils.ensure_output_dirs()
    # Log separat per etapa: etapele "trees" si "transformer" pot rula in paralel
    # (una pe CPU, cealalta pe GPU), iar doua procese care scriu in acelasi fisier
    # si-ar intercala liniile.
    log_path = (config.ADVERSARIAL_LOG_PATH if args.stage == "all"
                else config.ADVERSARIAL_DIR / f"adversarial_{args.stage}.log")
    logger = utils.setup_logging(log_path=log_path,
                                 logger_name=f"nids_adversarial_{args.stage}", mode="a")

    stages = STAGES if args.stage == "all" else [args.stage]
    state = runner.prepare(logger)

    if "trees" in stages:
        runner.train_trees(state, logger)
    if "transformer" in stages:
        runner.train_transformers(state, logger)
    if "evaluate" in stages:
        evaluated = runner.evaluate_runs(state, logger)
        cohorts = runner.build_cohorts(state, evaluated, logger)
        finalize(state, evaluated, cohorts, logger)

    logger.info(f"\nGata. Artefacte in {config.ADVERSARIAL_DIR}")


if __name__ == "__main__":
    main()
