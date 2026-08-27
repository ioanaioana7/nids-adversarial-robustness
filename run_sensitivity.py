"""
O4 — Analiza de sensibilitate (UNSW-NB15)
=========================================
Leaga rata de evaziune masurata in O3 de importanta caracteristicilor pe care
perturbarile le ating, si interpreteaza cele trei rezultate posibile. Nu
reantreneaza nimic: lucreaza pe modelele inghetate, prin calea determinista din
`src/inference.py`.

CE CALCULEAZA
  1. Importanta prin permutare pentru toate cele trei modele. Importantele native
     nu se pot compara (RF are impuritate, XGBoost "gain", Transformer atentie),
     iar intrebarea din O4 este tocmai o comparatie INTRE modele. Permutarea da o
     singura definitie, aplicata identic. Metrica principala e scaderea RATEI DE
     DETECTIE pe fluxurile de atac — exact marimea complementara evaziunii, deci
     cele doua sunt comensurabile.
  2. Cat din importanta totala sta in caracteristici pe care atacatorul le poate
     atinge, si cat concentreaza singura caracteristica dominanta.
  3. Corelatia de rang (Spearman) dintre masa de importanta atinsa de fiecare
     varianta si evaziunea pe care a produs-o.
  4. Destinatiile rezultatului (c): spre ce clase aluneca fluxurile confundate —
     harta granitelor fragile dintre clasele de atac. Plus daca fluxurile care
     evadeaza erau deja aproape de granita pe traficul curat.
  5. De ce diverg modelele pe clasa `Generic` (arborii rezista, Transformer-ul
     cedeaza), prin importanta restransa la acea clasa.

Cum se ruleaza:
    python run_sensitivity.py

Necesita O2 si O3 rulate in prealabil (feature_deltas.csv, evasion_summary.csv,
si rezultatele per flux din results/evasion/per_row/).

Rezultate in results/sensitivity/ si figuri in results/figures/sensitivity_*.png.
"""

from src import config, utils
from src.sensitivity import runner


def main():
    logger = utils.setup_logging(log_path=config.SENSITIVITY_LOG_PATH,
                                 logger_name="nids_sensitivity")
    runner.run(logger)


if __name__ == "__main__":
    main()
