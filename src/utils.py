"""Utilitare comune: seed global, creare directoare, logging, versiuni de librarii, I/O JSON."""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

from src import config


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seteaza seed-ul global (random, numpy) pentru executie deterministica.

    Parameters:
        seed (int): valoarea seed-ului. Modelele (RandomForest, XGBoost)
            primesc acelasi seed explicit prin random_state, in src.models.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs() -> None:
    """Creeaza toate directoarele de iesire definite in config.OUTPUT_DIRECTORIES."""
    for directory in config.OUTPUT_DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_logging(log_path=None, logger_name: str = "nids_baseline",
                  mode: str = "w") -> logging.Logger:
    """Configureaza logging-ul catre fisier si consola.

    Parameters:
        log_path (Path | str | None): fisierul de log; implicit config.LOG_PATH.
            O2 pasa config.PERTURBATION_LOG_PATH ca sa nu suprascrie log-ul O1.
        logger_name (str): numele logger-ului, pentru a izola handler-ele intre etape.
        mode (str): "w" suprascrie, "a" adauga. Antrenarea Transformer-ului
            foloseste "a", ca reluarea dintr-un checkpoint sa nu piarda istoricul
            rularii de peste noapte.

    Returns:
        logging.Logger: logger-ul principal al etapei.
    """
    ensure_output_dirs()
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path or config.LOG_PATH, mode=mode, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def library_versions() -> dict:
    """Colecteaza versiunile librariilor cheie folosite in pipeline, pentru reproducibilitate.

    Returns:
        dict: nume-librarie -> versiune.
    """
    import joblib
    import pandas as pd
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def save_json(data: dict, path) -> None:
    """Salveaza un dict ca JSON lizibil (indent=2).

    Parameters:
        data (dict): continutul de salvat.
        path (Path | str): fisierul de iesire; directorul parinte e creat daca lipseste.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
