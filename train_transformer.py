"""
FT-Transformer — antrenare standalone (UNSW-NB15, PyTorch)
==========================================================
Script de sine statator, gandit pentru o rulare lunga (peste noapte) pe CPU.
Se lanseaza si se lasa sa mearga; progresul se poate urmari in
results/transformer_training.log.

Aditiv fata de O1 (RF/XGBoost) si O2 (perturbare): nu le atinge si nu le
reantreneaza. Acelasi split oficial train/test, aceleasi features si etichete;
setul de validare e decupat stratificat din TRAIN, niciodata din test.

Diferenta de preprocesare fata de arbori e deliberata si trebuie mentionata in
lucrare: arborii primesc numericele nescalate ("passthrough"), pentru ca
folosesc doar ordinea valorilor; reteaua are nevoie de standardizare (medie 0,
varianta 1), ajustata exclusiv pe train. Nu e o incalcare de protocol, ci o
alegere de preprocesare per model, peste aceleasi date.

REZISTENTA LA INTRERUPERI
    Progresul se salveaza in models/transformer_checkpoint.pt la fiecare
    TRANSFORMER_CHECKPOINT_EVERY epoci SI la fiecare imbunatatire a scorului de
    validare. Daca rularea e intrerupta (reboot, inchidere accidentala), o
    relansare a aceleiasi comenzi continua exact de unde a ramas — inclusiv
    starile RNG, deci rezultatul ramane determinist.

UTILIZARE
    python train_transformer.py                # antreneaza (sau continua) + evalueaza
    python train_transformer.py --fresh        # ignora checkpoint-ul, o ia de la zero
    python train_transformer.py --eval-only    # doar artefacte, din modelul deja salvat
    python train_transformer.py --max-epochs 2 # rulare scurta de proba

REZULTATE
    models/transformer_baseline.pt                     ponderi + preprocesare
    models/transformer_checkpoint.pt                   stare de antrenare (reluare)
    results/metrics/transformer_metrics.csv            metrici agregate
    results/metrics/transformer_per_class.csv          precizie/recall/F1 per clasa
    results/metrics/confusion_matrix_transformer.csv   + PNG in results/figures/
    results/predictions/transformer_predictions.csv    predictii pe test curat
    results/predictions/baseline_probabilities_transformer.csv
    results/perturbation/eligible_flags_transformer.csv
    results/transformer_experiment_metadata.json
    results/transformer_training.log
"""

import argparse
import os

# Trebuie setat INAINTE de orice import care initializeaza cuBLAS, altfel
# torch.use_deterministic_algorithms(True) esueaza pe operatiile de matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src import config, utils              # noqa: E402
from src.transformer import runner         # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Antreneaza FT-Transformer-ul pe UNSW-NB15.")
    parser.add_argument("--max-epochs", type=int, default=config.TRANSFORMER_MAX_EPOCHS,
                        help=f"plafon de epoci (implicit {config.TRANSFORMER_MAX_EPOCHS})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignora checkpoint-ul existent si reia antrenarea de la zero")
    parser.add_argument("--eval-only", action="store_true",
                        help="sare peste antrenare; regenereaza artefactele din modelul salvat")
    return parser.parse_args()


def main():
    args = parse_args()
    # mode="a": o reluare nu trebuie sa stearga jurnalul rularii de peste noapte.
    logger = utils.setup_logging(log_path=config.TRANSFORMER_LOG_PATH,
                                 logger_name="nids_transformer",
                                 mode="w" if args.fresh else "a")
    summary = runner.run(logger, max_epochs=args.max_epochs,
                         resume=not args.fresh, eval_only=args.eval_only)
    logger.info(f"\nGata. macro F1 = {summary['macro_f1']:.4f}, "
                f"weighted F1 = {summary['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
