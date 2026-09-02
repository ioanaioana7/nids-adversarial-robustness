"""Antrenarea FT-Transformer-ului: determinism, dezechilibru de clase, early stopping."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src import config


def seed_everything(seed: int = config.RANDOM_SEED,
                    deterministic: bool = config.TRANSFORMER_DETERMINISTIC) -> None:
    """Fixeaza toate sursele de aleatoriu, inclusiv cele din torch.

    Parameters:
        seed (int): seed-ul comun, din config.RANDOM_SEED.
        deterministic (bool): forteaza algoritmi deterministi si pe GPU. Fara el,
            backward-ul de nn.Embedding foloseste atomicAdd si doua rulari cu
            acelasi seed pot diferi.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Trebuie setat inainte de initializarea cuBLAS, altfel e ignorat.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", config.CUBLAS_WORKSPACE_CONFIG)
        torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    """Seed determinist pentru workerii DataLoader-ului."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool, seed: int = config.RANDOM_SEED) -> DataLoader:
    """Construieste un DataLoader determinist peste tensorii deja preprocesati.

    Parameters:
        x_num (np.ndarray): numerice standardizate [n, n_num].
        x_cat (np.ndarray): indici categoriali [n, n_cat].
        y (np.ndarray): etichete intregi [n].
        batch_size (int): dimensiunea lotului.
        shuffle (bool): amestecare (doar pentru train).
        seed (int): seed-ul generatorului de amestecare.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = TensorDataset(torch.from_numpy(x_num),
                            torch.from_numpy(x_cat),
                            torch.from_numpy(y.astype(np.int64)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.TRANSFORMER_DATALOADER_WORKERS,
                      generator=generator, worker_init_fn=_seed_worker)


def class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Ponderi de clasa dupa formula "balanced" din sklearn: n / (k * count_c).

    Echivalentul pentru retea al lui class_weight="balanced" folosit de arbori.
    Fara ele, clasele minoritare (Worms=130, Shellcode, Analysis, Backdoor) sunt
    ignorate de model.

    Parameters:
        y_train (np.ndarray): etichete intregi de antrenare.
        n_classes (int): numarul total de clase.

    Returns:
        torch.Tensor: ponderi [n_classes], float32.
    """
    present = np.unique(y_train)
    computed = compute_class_weight("balanced", classes=present, y=y_train)
    weights = np.ones(n_classes, dtype=np.float32)
    for cls, weight in zip(present, computed):
        weights[cls] = weight
    return torch.from_numpy(weights)


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Predictii si macro-F1 pe un split, in modul eval.

    Returns:
        tuple: (macro_f1, y_true, y_pred).
    """
    model.eval()
    true_batches, pred_batches = [], []
    for num, cat, target in loader:
        logits = model(num.to(device), cat.to(device))
        pred_batches.append(logits.argmax(dim=1).cpu().numpy())
        true_batches.append(target.numpy())
    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(pred_batches)
    return f1_score(y_true, y_pred, average="macro"), y_true, y_pred


def save_checkpoint(path, epoch: int, model: nn.Module, optimizer, train_loader: DataLoader,
                    best_score: float, best_epoch: int, best_state: dict, history: list) -> None:
    """Salveaza starea completa de antrenare, ca reluarea sa fie bit-exacta.

    Include starile RNG (Python/NumPy/torch) si starea generatorului DataLoader-ului,
    fara de care ordinea loturilor dupa reluare ar reincepe de la epoca 1 in loc sa
    continue secventa.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score_smoothed": best_score,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": train_loader.generator.get_state(),
        },
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }, path)


def load_checkpoint(path, model: nn.Module, optimizer, train_loader: DataLoader, device,
                    logger=None):
    """Reia starea salvata de save_checkpoint.

    Reluarea e bit-exacta doar pe ACELASI tip de dispozitiv pe care s-a salvat:
    aritmetica in virgula mobila difera intre CPU si GPU, iar starea RNG a
    fiecaruia e separata. La schimbarea dispozitivului se avertizeaza explicit.

    Returns:
        tuple: (epoca urmatoare, best_score netezit, best_epoch, best_state, history).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    saved_device = state.get("device_type", "cpu")
    if logger is not None and saved_device != device.type:
        logger.info(f"  [atentie] checkpoint salvat pe '{saved_device}', se reia pe "
                    f"'{device.type}': continuarea NU va fi bit-exacta. Pentru un "
                    f"rezultat curat foloseste --fresh.")

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available() and saved_device == "cuda":
        torch.cuda.set_rng_state_all(rng["cuda"])
    train_loader.generator.set_state(rng["loader"])

    return (state["epoch"] + 1, state["best_score_smoothed"], state["best_epoch"],
            state["best_state_dict"], state["history"])


def _format_eta(seconds: float) -> str:
    """Durata in format h/m, pentru linia de progres."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          weights: torch.Tensor, device: torch.device, logger,
          max_epochs: int = config.TRANSFORMER_MAX_EPOCHS,
          patience: int = config.TRANSFORMER_EARLY_STOPPING_PATIENCE,
          checkpoint_path=config.TRANSFORMER_CHECKPOINT_PATH,
          checkpoint_every: int = config.TRANSFORMER_CHECKPOINT_EVERY,
          resume: bool = True,
          window: int = config.TRANSFORMER_SELECTION_WINDOW,
          loader_factory=None) -> tuple[dict, list]:
    """Antreneaza cu early stopping pe macro-F1 de validare, cu checkpointing.

    Macro-F1 (nu acuratetea) e criteriul, pentru ca setul e puternic dezechilibrat:
    un model care ignora complet clasele rare ar avea acuratete buna si macro-F1 slab.

    Progresul se salveaza la fiecare `checkpoint_every` epoci SI la fiecare
    imbunatatire a scorului de validare, deci o intrerupere (reboot, inchidere
    accidentala) costa cel mult cateva epoci, nu toata rularea.

    Parameters:
        model (nn.Module): reteaua de antrenat.
        train_loader (DataLoader): loturile de antrenare.
        val_loader (DataLoader): loturile de validare (decupate din TRAIN).
        weights (torch.Tensor): ponderile de clasa pentru CrossEntropyLoss.
        device (torch.device): dispozitivul de calcul.
        logger (logging.Logger): logger.
        max_epochs (int): numarul maxim de epoci.
        patience (int): cate epoci fara imbunatatire inainte de oprire.
        checkpoint_path (Path): unde se salveaza starea de antrenare.
        checkpoint_every (int): interval (in epoci) intre salvarile periodice.
        resume (bool): daca True si exista un checkpoint, continua de acolo.
        loader_factory (callable | None): daca e dat, se apeleaza cu numarul
            epocii si returneaza DataLoader-ul acelei epoci. Exista pentru O6:
            augmentarea adversariala reesantioneaza variantele la fiecare epoca,
            deci loturile nu mai pot fi fixate o singura data inainte de bucla.
            Cu None (implicit, cazul O1) se foloseste train_loader la fiecare
            epoca, iar comportamentul e neschimbat.

    Returns:
        tuple[dict, list]: (rezumatul antrenarii, istoricul per epoca).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRANSFORMER_LR,
                                  weight_decay=config.TRANSFORMER_WEIGHT_DECAY)

    start_epoch, best_score, best_epoch, best_state, history = 1, -1.0, -1, None, []
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        start_epoch, best_score, best_epoch, best_state, history = load_checkpoint(
            checkpoint_path, model, optimizer, train_loader, device, logger)
        logger.info(f"  [reluare] checkpoint gasit: continui de la epoca {start_epoch} "
                    f"(cel mai bun pana acum: epoca {best_epoch}, scor netezit={best_score:.4f})")

    if start_epoch > max_epochs:
        logger.info(f"  antrenarea era deja completa ({start_epoch - 1}/{max_epochs} epoci)")
    epoch_times = [h["seconds"] for h in history]

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        started = time.time()
        epoch_loader = loader_factory(epoch) if loader_factory is not None else train_loader
        total_loss, n_batches = 0.0, 0
        for num, cat, target in epoch_loader:
            num, cat, target = num.to(device), cat.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(num, cat), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.TRANSFORMER_GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_f1, _, _ = evaluate_split(model, val_loader, device)
        elapsed = time.time() - started
        epoch_times.append(elapsed)
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_batches, 1),
                        "val_macro_f1": val_f1, "seconds": elapsed})

        # Scorul de decizie e media mobila a ultimelor `window` epoci, nu valoarea
        # epocii curente: o singura epoca norocoasa nu mai poate nici sa fie salvata
        # ca model final, nici sa opreasca antrenarea.
        recent = [h["val_macro_f1"] for h in history[-window:]]
        smoothed = float(np.mean(recent))
        history[-1]["val_macro_f1_smoothed"] = smoothed

        improved = smoothed > best_score
        if improved:
            best_score, best_epoch = smoothed, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        eta = _format_eta(float(np.mean(epoch_times[-5:])) * (max_epochs - epoch))
        logger.info(f"  epoca {epoch:>3}/{max_epochs}  loss={total_loss / max(n_batches, 1):.4f}  "
                    f"val_macroF1={val_f1:.4f}  netezit={smoothed:.4f}  {elapsed:.0f}s  ETA<={eta}"
                    f"{'  <-- cel mai bun' if improved else ''}")

        if checkpoint_path is not None and (improved or epoch % checkpoint_every == 0):
            save_checkpoint(checkpoint_path, epoch, model, optimizer, train_loader,
                            best_score, best_epoch, best_state, history)

        if epoch - best_epoch >= patience:
            logger.info(f"  early stopping: fara imbunatatire de {patience} epoci "
                        f"(cel mai bun: epoca {best_epoch}, scor netezit={best_score:.4f})")
            break

    if best_state is None:
        raise RuntimeError("nicio epoca antrenata si niciun checkpoint valid gasit")

    model.load_state_dict(best_state)
    best_raw = next((h["val_macro_f1"] for h in history if h["epoch"] == best_epoch), float("nan"))
    logger.info(f"  restaurat checkpoint-ul de la epoca {best_epoch} "
                f"(scor netezit={best_score:.4f}, val_macroF1 brut={best_raw:.4f})")
    return {"best_epoch": best_epoch, "best_val_macro_f1_smoothed": best_score,
            "best_val_macro_f1_raw": best_raw, "selection_window": window,
            "epochs_run": len(history),
            "total_training_seconds": float(sum(epoch_times))}, history
