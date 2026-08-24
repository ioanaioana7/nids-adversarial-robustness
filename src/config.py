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

# Predictie determinista cu modelele inghetate.
# RandomForest-ul antrenat are 44 de randuri pe muchie de cutit in setul de atac
# (43 la egalitate exacta, unul cu diferenta de 1 ULP: sample_id=49676, unde
# Exploits=0.392499999999999905 si Reconnaissance=0.392500000000000016). Cu
# n_jobs=-1, ordinea de insumare a voturilor arborilor variaza intre rulari si
# argmax-ul poate oscila. Predictia pe un singur fir elimina complet variatia,
# la un cost neglijabil (1.2s vs 0.4s pentru 45.332 de randuri).
DETERMINISTIC_INFERENCE = True
DETERMINISTIC_INFERENCE_N_JOBS = 1

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
# FT-Transformer (al treilea clasificator, PyTorch)
# --------------------------------------------------------------------------
# Protocol: acelasi split oficial train/test ca la arbori. Setul de validare
# (pentru early stopping) e decupat STRATIFICAT din TRAIN, niciodata din test.
TRANSFORMER_VAL_FRACTION = 0.15

# Tokenizer + arhitectura (Gorishniy et al., NeurIPS 2021)
TRANSFORMER_D_TOKEN = 192
TRANSFORMER_N_BLOCKS = 3
TRANSFORMER_N_HEADS = 8
TRANSFORMER_ATTN_DROPOUT = 0.2
TRANSFORMER_FFN_DROPOUT = 0.1
TRANSFORMER_RESIDUAL_DROPOUT = 0.0
TRANSFORMER_FFN_HIDDEN_MULTIPLIER = 4 / 3   # hidden = d_token * 4/3 -> 256

# Optimizare
TRANSFORMER_LR = 1e-4
TRANSFORMER_WEIGHT_DECAY = 1e-5
TRANSFORMER_BATCH_SIZE = 512
TRANSFORMER_EVAL_BATCH_SIZE = 2048
TRANSFORMER_MAX_EPOCHS = 100
TRANSFORMER_EARLY_STOPPING_PATIENCE = 10
TRANSFORMER_GRAD_CLIP_NORM = 1.0

# Determinism: 0 workers evita nedeterminismul de multiprocessing pe Windows.
TRANSFORMER_DATALOADER_WORKERS = 0
# Pe GPU, backward-ul de nn.Embedding foloseste atomicAdd, care e nedeterminist.
# torch.use_deterministic_algorithms(True) il forteaza pe varianta determinista.
# Cost masurat: +7% timp de antrenare — merita, determinismul fiind criteriu de
# verificare. Necesita CUBLAS_WORKSPACE_CONFIG=:4096:8 (setat automat).
TRANSFORMER_DETERMINISTIC = True
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Checkpointing: antrenarea dureaza ore pe CPU, deci progresul se salveaza
# periodic SI la fiecare imbunatatire, ca o intrerupere sa nu coste tot efortul.
TRANSFORMER_CHECKPOINT_EVERY = 10           # epoci intre salvarile periodice

# Verificari
TRANSFORMER_RELOAD_CHECK_ROWS = 100         # oglindeste validate_saved_model din O1

# Bucket pentru categoriile necunoscute/rare, aliniat cu RareCategoryGrouper.
TRANSFORMER_UNKNOWN_CATEGORY = OTHER_CATEGORY_LABEL

# --------------------------------------------------------------------------
# O2 — perturbare adversariala
# --------------------------------------------------------------------------
# Taxonomia caracteristicilor dupa cine le controleaza in realitate.
# Atacatorul controleaza doar propriile pachete; features "dst" sunt generate
# de victima, deci nu pot fi modificate direct.
SOURCE_CONTROLLED = ["sttl", "sbytes", "spkts", "dur", "sinpkt", "sjit", "swin", "stcpb"]

# Agregate pe fereastra de 100 de conexiuni: atacatorul le poate reduce
# incetinind atacul, dar nu le poate creste arbitrar per-flux.
CONNECTION_RATE_FEATURES = [
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

# Recalculate prin propagare, niciodata setate direct.
DERIVED_FEATURES = ["smean", "sload", "rate", "sinpkt", "ct_state_ttl"]

# Generate de victima: trebuie sa ramana neschimbate...
VICTIM_CONTROLLED = [
    "dttl", "dbytes", "dpkts", "dwin", "dtcpb", "dmean", "dloss", "response_body_len",
]
# ...cu excepatia celor antrenate cauzal de o schimbare de `dur` (daca atacatorul
# intarzie, raspunsurile victimei se intind si ele in timp).
CAUSALLY_FORCED_BY_DUR = ["dload", "dinpkt", "djit"]

# Definesc identitatea atacului: modificarea lor ar schimba atacul insusi.
IDENTITY_DEFINING = ["proto", "service", "state", "is_sm_ips_ports"]

# Constrangeri de realism fizic
MTU_BYTES = 1500                          # plafon pentru dimensiunea medie a pachetului
MIN_REALISTIC_TTL = 16                    # sub acest prag pachetul risca sa nu ajunga la tinta
MAX_TTL = 255
MIN_CONNECTION_COUNT = 1                  # minimul observat pentru contoarele ct_*

# Niveluri de perturbare (nivelul 0 = identitate, control)
# Doar valori sttl efectiv observate in captura (128 nu apare niciodata).
TTL_LEVELS = [64, 62, 31]
NORMAL_TTL_TARGET = 31                    # modul sttl al traficului normal (70.5%)
NORMAL_DTTL_TARGET = 29                   # modul dttl al traficului normal
PADDING_LEVELS = [0.10, 0.25, 0.50, 1.00]     # fractie adaugata la sbytes
TIMING_LEVELS = [1.5, 2.0, 5.0, 10.0]         # factor de dilatare a duratei
CONNECTION_RATE_LEVELS = [0.75, 0.50, 0.25, 0.0]   # 0.0 => coboara la minim
COMBINED_LEVELS = [1, 2, 3, 4]

# Politica pentru ct_state_ttl: nu este o functie de intervale TTL (sttl=62->2
# dar sttl=63->0), ci un contor pe fereastra glisanta, imposibil de recalculat
# din inregistrari de flux. Emitem ambele margini:
#   "hold"  -> ct_state_ttl neschimbat  (conservator, subestimeaza evaziunea)
#   "mimic" -> semnatura traficului normal pentru sttl-ul tinta (optimist)
CT_STATE_TTL_POLICIES = ["hold", "mimic"]
CT_STATE_TTL_FALLBACK = 0                 # cand nici lookup-ul, nici modul normal nu exista

# Toleranta pentru verificarea identitatilor de propagare
PROPAGATION_RTOL = 1e-6

PERTURBATION_PERSIST_VARIANTS = False     # True => salveaza matricile complete
PERTURBATION_SAMPLE_ROWS = 200            # randuri per variant salvate pentru inspectie

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

PERTURBATION_DIR = RESULTS_DIR / "perturbation"
PERTURBATION_SAMPLES_DIR = PERTURBATION_DIR / "samples"
PERTURBATION_VARIANTS_DIR = PERTURBATION_DIR / "variants"
PERTURBATION_LOG_PATH = PERTURBATION_DIR / "perturbation.log"

RF_MODEL_PATH = MODELS_DIR / "rf_baseline.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_baseline.joblib"

# FT-Transformer. Log separat, ca sa nu suprascrie training.log de la O1.
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_baseline.pt"
TRANSFORMER_CHECKPOINT_PATH = MODELS_DIR / "transformer_checkpoint.pt"
TRANSFORMER_LOG_PATH = RESULTS_DIR / "transformer_training.log"
TRANSFORMER_METADATA_PATH = RESULTS_DIR / "transformer_experiment_metadata.json"
# Fisiere PARALELE per model: nu modificam artefactele detinute de O1/O2
# (baseline_predictions.csv, eligible_flags.csv), care sunt regenerate integral
# la o rerulare a lui main.py / run_perturbation.py.
TRANSFORMER_PREDICTIONS_PATH = PREDICTIONS_DIR / "transformer_predictions.csv"
TRANSFORMER_ELIGIBLE_FLAGS_PATH = PERTURBATION_DIR / "eligible_flags_transformer.csv"

OUTPUT_DIRECTORIES = [
    MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    FEATURE_IMPORTANCE_DIR, EDA_DIR, PREDICTIONS_DIR,
    PERTURBATION_DIR, PERTURBATION_SAMPLES_DIR,
]
