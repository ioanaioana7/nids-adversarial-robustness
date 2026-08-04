"""Configurare centralizata pentru pipeline-ul baseline NIDS (UNSW-NB15).

Nicio valoare configurabila (cai, coloane, hiperparametri, directoare) nu
trebuie sa apara hardcodata in afara acestui modul.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Cai catre date
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Resources" / "CSV Files" / "Training and Testing Sets"
TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

# --------------------------------------------------------------------------
# Coloane
# --------------------------------------------------------------------------
ID_COL = "id"
TARGET = "attack_cat"                     # eticheta multi-clasa
LABEL_COL = "label"                       # eticheta binara (normal/atac)
DROP_COLUMNS = []                         # coloane suplimentare de eliminat la incarcare
CATEGORICAL_COLUMNS = ["proto", "service", "state"]
TTL_FEATURES = ["sttl", "dttl"]           # relevante pentru partea de perturbare (O2)

# --------------------------------------------------------------------------
# Preprocesare
# --------------------------------------------------------------------------
MIN_CATEGORY_FREQUENCY = 50               # prag pentru gruparea categoriilor rare
OTHER_CATEGORY_LABEL = "other"

# --------------------------------------------------------------------------
# Reproducibilitate
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1

# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.9
XGB_TREE_METHOD = "hist"
XGB_N_JOBS = -1

# --------------------------------------------------------------------------
# SHAP (O4 prep)
# --------------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 500

# --------------------------------------------------------------------------
# Directoare de iesire
# --------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
FEATURE_IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"
EDA_DIR = RESULTS_DIR / "eda"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOG_PATH = RESULTS_DIR / "training.log"

OUTPUT_DIRECTORIES = [
    MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    FEATURE_IMPORTANCE_DIR, EDA_DIR, PREDICTIONS_DIR,
]
