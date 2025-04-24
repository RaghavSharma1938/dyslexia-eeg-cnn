# src/evaluate.py
# ---------------------------------------------------------------------------
# Evaluate the dyslexia-EEG CNN in one go:
#   • window-level accuracy / confusion / ROC-AUC
#   • subject-level accuracy (majority vote)
#   • optional LOSO cross-validation with --loso
# ---------------------------------------------------------------------------

import argparse, pathlib, re, json, collections, sys, warnings
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import pytorch_lightning as pl

# --- project imports ------------------------------------------------------- #
sys.path.append(str(pathlib.Path(__file__).parents[1]))  # add repo root to PYTHONPATH
from src.data.dataset import DyslexiaDataModule
from src.models.cnn import SimpleCNN
from train import Lit  # Lightning wrapper used during training

# --------------------------------------------------------------------------- #
def load_best_checkpoint(checkpoint_dir="checkpoints") -> pathlib.Path:
    ckpts = list(pathlib.Path(checkpoint_dir).glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    best = max(
        ckpts,
        key=lambda p: float(re.search(r"val([0-9.]+)", p.stem).group(1)),
    )
    return best


def window_metrics(model: torch.nn.Module, dm: DyslexiaDataModule):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            logits = model(x)
            y_true += y.tolist()
            y_pred += logits.argmax(1).tolist()
            y_prob += torch.softmax(logits, 1)[:, 1].tolist()

    print("\n── Window-level metrics ───────────────────────────────")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # ROC needs >1 class
        try:
            print("ROC-AUC  :", roc_auc_score(y_true, y_prob))
        except ValueError:
            print("ROC-AUC  : undefined (only one class present)")
    print(classification_report(y_true, y_pred, digits=3))


def subject_metrics(model: torch.nn.Module, dm: DyslexiaDataModule):
    model.eval()
    votes = collections.defaultdict(list)
    truth = {}

    # iterate raw files for clarity
    proc_dir = pathlib.Path("data/processed")
    for f in proc_dir.glob("*_X.npy"):
        subj = f.stem[:3]
        if subj not in dm.val_subj:
            continue  # only validation subjects
        x = torch.tensor(np.load(f)).float()
        preds = model(x).argmax(1).tolist()
        votes[subj].extend(preds)
        truth[subj] = 0 if subj.startswith("h") else 1

    subj_pred = {s: round(sum(v) / len(v)) for s, v in votes.items()}
    y_true = list(truth.values())
    y_pred = [subj_pred[s] for s in truth.keys()]

    print("\n── Subject-level metrics (majority vote) ──────────────")
    print("Subjects :", len(truth))
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))


def run_single_eval():
    ckpt = load_best_checkpoint()
    dm = DyslexiaDataModule(batch=256)
    dm.setup()
    model = Lit.load_from_checkpoint(ckpt, n_ch=dm.n_channels).model.cpu()

    print(f"Loaded checkpoint: {ckpt.name}")
    print(f"Validation subjects: {dm.val_subj}\n")

    window_metrics(model, dm)
    subject_metrics(model, dm)


# --------------------------------------------------------------------------- #
def loso_cv():
    proc = pathlib.Path("data/processed")
    all_subjects = sorted({f.stem[:3] for f in proc.glob("*_X.npy")})
    if len(all_subjects) < 3:
        raise RuntimeError("Need at least 3 subjects for LOSO-CV")

    scores = {}
    for hold in all_subjects:
        print(f"\n===== LOSO fold: hold-out {hold} =====")
        dm = DyslexiaDataModule(batch=256)
        dm.setup()
        dm.val_subj = [hold]
        dm.train_subj = [s for s in all_subjects if s != hold]
        dm.train_files = [
            f
            for f in proc.glob("*_X.npy")
            if f.stem[:3] in dm.train_subj
        ]
        dm.val_files = [
            f
            for f in proc.glob("*_X.npy")
            if f.stem[:3] in dm.val_subj
        ]

        # fresh model every fold
        model = SimpleCNN(dm.n_channels)
        lit = Lit(dm.n_channels)
        lit.model = model
        trainer = pl.Trainer(max_epochs=30, logger=False, enable_checkpointing=False)
        trainer.fit(lit, datamodule=dm)

        # evaluate subject-level
        lit.eval()
        votes, truth = [], []
        for f in dm.val_files:
            x = torch.tensor(np.load(f)).float()
            pred = lit.model(x).argmax(1).mode()[0].item()
            votes.append(pred)
            truth.append(0 if f.stem.startswith("h") else 1)
        acc = accuracy_score(truth, votes)
        scores[hold] = acc
        print(f"Subject accuracy for {hold}: {acc:.3f}")

    print("\n===== LOSO summary =====")
    print(json.dumps(scores, indent=2))
    print("Mean subject accuracy:", np.mean(list(scores.values())))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--loso",
        action="store_true",
        help="run leave-one-subject-out cross-validation instead of single eval",
    )
    args = p.parse_args()

    pl.seed_everything(42, workers=True)

    if args.loso:
        loso_cv()
    else:
        run_single_eval()
