"""
Dataset + Lightning DataModule for the dyslexia-EEG project
-----------------------------------------------------------

• Loads paired   *_X.npy   *_y.npy   files produced by build_numpy.py  
• Creates **subject-wise** train/val splits  
     – 80 % of subjects → train, 20 % → validation  
     – keeps *both* classes (normal / dyslexic) in each split  
• Keeps num_workers = 0 to avoid Windows pickle issues
"""

from __future__ import annotations
import pathlib, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl


# ─────────────────────────────────────────────────────────────────────────────
#  tiny wrapper around one subject’s window tensors
# ─────────────────────────────────────────────────────────────────────────────
class NPYPair(Dataset):
    def __init__(self, x_file: pathlib.Path, y_file: pathlib.Path):
        self.X = np.load(x_file, mmap_mode="r").astype(np.float32)
        self.y = np.load(y_file, mmap_mode="r").astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),       # (C, T) float32
            torch.tensor(int(self.y[idx])),      # int64 label
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Lightning DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DyslexiaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        proc_dir: str | pathlib.Path = "data/processed",
        batch: int = 64,
    ):
        super().__init__()
        self.proc = pathlib.Path(proc_dir)
        self.batch = batch
        self.num_workers = 0  # Windows → keep workers = 0 (no pickle issues)

    # ---------- internal helpers ------------------------------------------- #
    def _group_by_subject(self) -> dict[str, list[pathlib.Path]]:
        """
        Return {subject_id: [file_X.npy, …]} mapping.
        Subject ID = first 3 chars (h01 / s14) — adjust if your pattern differs.
        """
        mapping: dict[str, list[pathlib.Path]] = {}
        for f in self.proc.glob("*_X.npy"):
            mapping.setdefault(f.stem[:3], []).append(f)
        return mapping

    def _build_loader(self, files: list[pathlib.Path], shuffle: bool) -> DataLoader:
        datasets = [
            NPYPair(f, f.with_name(f.name.replace("_X", "_y"))) for f in files
        ]
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    # ---------- Lightning hooks ------------------------------------------- #
    def setup(self, stage: str | None = None):
        subj2files = self._group_by_subject()
        if not subj2files:
            raise RuntimeError(
                f"No *_X.npy files found in {self.proc}. "
                "Run  python src/data/build_numpy.py  first."
            )

        subjects = list(subj2files.keys())
        if len(subjects) < 3:
            raise RuntimeError(
                f"Need at least 3 subjects for a split — found only {len(subjects)}."
            )

        # ── balanced 20 % validation set (alternate normal / dyslexic) ──
        normals  = [s for s in subjects if s.startswith("h")]
        dyslexic = [s for s in subjects if s.startswith("s")]
        random.seed(42)
        random.shuffle(normals)
        random.shuffle(dyslexic)

        n_val_target = max(1, int(0.2 * len(subjects)))
        val_subj: list[str] = []
        while len(val_subj) < n_val_target and (normals or dyslexic):
            if normals:
                val_subj.append(normals.pop())
            if len(val_subj) < n_val_target and dyslexic:
                val_subj.append(dyslexic.pop())

        self.val_subj   = val_subj
        self.train_subj = [s for s in subjects if s not in val_subj]

        self.train_files = [f for s in self.train_subj for f in subj2files[s]]
        self.val_files   = [f for s in self.val_subj   for f in subj2files[s]]

        if not self.train_files or not self.val_files:
            raise RuntimeError(
                "Split produced an empty train or val set — "
                "check filename prefixes (should start with h / s)."
            )

    def train_dataloader(self):
        return self._build_loader(self.train_files, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_files, shuffle=False)

    # ---------- convenience property -------------------------------------- #
    @property
    def n_channels(self) -> int:
        """Infer channel count from any sample tensor."""
        sample = next(self.proc.glob("*_X.npy"), None)
        if sample is None:
            raise RuntimeError("No *_X.npy present; run build_numpy.py first.")
        return np.load(sample, mmap_mode="r").shape[1]
