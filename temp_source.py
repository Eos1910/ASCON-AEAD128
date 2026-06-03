# CELL TYPE: markdown
# ASCON-AEAD-128 CNN256 Top-k Model Training, All 16 Bytes

This notebook trains byte-wise 256-class key-byte classifiers for the ASCON-AEAD-128 tight-trigger dataset.

It is prepared for your usual Windows PC with CUDA/GPU training.

Folder policy used in this notebook:

```text
Traces only:
C:\Users\thetp\Downloads\ASCON_FYP\03_Data

Model training results:
C:\Users\thetp\Downloads\ASCON_FYP\04_Results

Model checkpoints:
C:\Users\thetp\Downloads\ASCON_FYP\05_Models
```

Main outputs:

```text
Single trace: Top1, Top2, Top4, Top8, Top16, Top32
Grouped trace: grouped_n1, grouped_n2, grouped_n4, grouped_n8, grouped_n12, grouped_n16
Per-byte model checkpoints
Per-byte summaries
All-byte summary
Candidate-space reduction summary
```

# CELL TYPE: markdown
## 1. Environment and CUDA Check

# CELL TYPE: code
import sys
import platform
import torch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.platform())
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
else:
    print("WARNING: CUDA is not available. Training will run on CPU and will be much slower.")

# CELL TYPE: markdown
## 2. Imports

# CELL TYPE: code
from pathlib import Path
from datetime import datetime
import math
import random
import time
import gc
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)

# CELL TYPE: markdown
## 3. Configuration

# CELL TYPE: code
# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\thetp\Downloads\ASCON_FYP")

# Traces only
DATA_BASE = PROJECT_ROOT / r"03_Data\tight_trigger_keybyte_auto30k"

# Results and model checkpoints
RESULTS_BASE = PROJECT_ROOT / r"04_Results\ascon_cnn256_topk_training"
MODELS_BASE = PROJECT_ROOT / r"05_Models\ascon_cnn256_topk_training"

# If None, the notebook auto-detects the newest dataset folder under DATA_BASE.
# You can also set it manually, for example:
# DATASET_RUN_NAME = "ascon_tighttrigger_30k_per_byte_20260526_011623"
DATASET_RUN_NAME = None

# If None, a new timestamped training run folder is created.
# If you want to resume a specific previous run, set it manually.
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")

RESULTS_ROOT = RESULTS_BASE / RUN_ID
MODELS_ROOT = MODELS_BASE / RUN_ID
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_BASE:", DATA_BASE)
print("RESULTS_ROOT:", RESULTS_ROOT)
print("MODELS_ROOT:", MODELS_ROOT)

# ============================================================
# Byte selection
# ============================================================
RUN_ALL_BYTES = True
TARGET_BYTE = 0
BYTES_TO_TRAIN = list(range(16)) if RUN_ALL_BYTES else [TARGET_BYTE]

# If True, skip byte if its per-byte summary already exists in this RUN_ID.
SKIP_COMPLETED_BYTES = True

# ============================================================
# Top-k and grouped settings
# ============================================================
TOPK_VALUES = (1, 2, 4, 8, 16, 32)
GROUP_SIZES = (1, 2, 4, 8, 12, 16)

# ============================================================
# Data split settings
# ============================================================
SEED = 42
TEST_SIZE_TOTAL = 0.30       # 70% train, 30% temporary
VAL_SIZE_FROM_TEMP = 0.50    # 15% val, 15% test

# ============================================================
# Training settings
# ============================================================
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
EARLY_STOPPING_PATIENCE = 15
GRAD_CLIP_NORM = 1.0

# If RTX 3070 8GB runs out of memory, change BATCH_SIZE to 64.
# BATCH_SIZE = 64

# ============================================================
# Augmentation settings
# ============================================================
USE_AUGMENTATION = True
AUG_SHIFT_MAX = 4
AUG_NOISE_STD = 0.005
AUG_SCALE_STD = 0.02

# ============================================================
# Saving settings
# ============================================================
SAVE_SPLIT_MAPPING = True
SAVE_TOP32_PREDICTIONS = True
SAVE_CONFIG_COPY = True

# ============================================================
# Device
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if device == "cuda" else False

print("Device:", device)
print("Bytes to train:", BYTES_TO_TRAIN)

# CELL TYPE: markdown
## 4. Locate and Verify Dataset Folder

# CELL TYPE: code
def find_byte_files(byte_dir: Path, target_byte: int):
    """Return trace/meta files using the common naming patterns."""
    patterns_trace = [
        f"byte{target_byte:02d}_chunk_*_traces.npz",
        f"ascon_byte{target_byte:02d}_chunk_*_traces.npz",
        "*_traces.npz",
    ]
    patterns_meta = [
        f"byte{target_byte:02d}_chunk_*_metadata.csv",
        f"ascon_byte{target_byte:02d}_chunk_*_metadata.csv",
        "*_metadata.csv",
    ]

    trace_files = []
    meta_files = []

    for pat in patterns_trace:
        trace_files = sorted(byte_dir.glob(pat))
        if trace_files:
            break

    for pat in patterns_meta:
        meta_files = sorted(byte_dir.glob(pat))
        if meta_files:
            break

    return trace_files, meta_files


def dataset_score(folder: Path):
    """Score a possible ASCON dataset folder.

    Supports both layouts:
    1. DATA_BASE / ascon_tighttrigger_30k_per_byte_YYYYMMDD_HHMMSS / byte_00 / files
    2. DATA_BASE / byte_00 / files
    """
    if not folder.exists() or not folder.is_dir():
        return None

    if "PLACEHOLDER" in folder.name.upper():
        return None

    total_trace_files = 0
    total_meta_files = 0
    total_meta_rows = 0
    bytes_with_data = 0

    for b in range(16):
        byte_dir = folder / f"byte_{b:02d}"
        if not byte_dir.exists() or not byte_dir.is_dir():
            continue

        trace_files, meta_files = find_byte_files(byte_dir, b)
        total_trace_files += len(trace_files)
        total_meta_files += len(meta_files)

        byte_meta_rows = 0
        for mf in meta_files:
            try:
                byte_meta_rows += len(pd.read_csv(mf))
            except Exception:
                pass

        total_meta_rows += byte_meta_rows

        if len(trace_files) > 0 and len(meta_files) > 0 and byte_meta_rows > 0:
            bytes_with_data += 1

    return {
        "folder": folder,
        "bytes_with_data": bytes_with_data,
        "total_trace_files": total_trace_files,
        "total_meta_files": total_meta_files,
        "total_meta_rows": total_meta_rows,
    }


def autodetect_dataset_root():
    candidates = []

    # Candidate 1: DATA_BASE itself, if byte_00 is directly inside it.
    score = dataset_score(DATA_BASE)
    if score is not None:
        candidates.append(score)

    # Candidate 2: one-level child run folders.
    for p in sorted(DATA_BASE.iterdir() if DATA_BASE.exists() else []):
        score = dataset_score(p)
        if score is not None:
            candidates.append(score)

    if not candidates:
        raise FileNotFoundError(
            f"No valid ASCON dataset found under {DATA_BASE}.\n"
            "Expected either:\n"
            f"  {DATA_BASE}\\byte_00\\...\n"
            "or:\n"
            f"  {DATA_BASE}\\ascon_tighttrigger_30k_per_byte_YYYYMMDD_HHMMSS\\byte_00\\..."
        )

    candidate_df = pd.DataFrame([
        {
            "folder": str(c["folder"]),
            "bytes_with_data": c["bytes_with_data"],
            "total_trace_files": c["total_trace_files"],
            "total_meta_files": c["total_meta_files"],
            "total_meta_rows": c["total_meta_rows"],
        }
        for c in candidates
    ]).sort_values(["bytes_with_data", "total_meta_rows", "total_trace_files"], ascending=False)

    print("Dataset candidates found:")
    display(candidate_df)

    best_folder = Path(candidate_df.iloc[0]["folder"])
    return best_folder


# If DATASET_RUN_NAME is None, auto-detect the real dataset and ignore PLACEHOLDER folders.
# If your byte_00...byte_15 folders are directly inside DATA_BASE, this will use DATA_BASE.
# If they are inside a timestamped run folder, this will use that run folder.
if DATASET_RUN_NAME is None:
    DATASET_ROOT = autodetect_dataset_root()
else:
    DATASET_ROOT = DATA_BASE / DATASET_RUN_NAME

print("Using dataset root:", DATASET_ROOT)

# Quick file count verification.
rows = []
for b in range(16):
    byte_dir = DATASET_ROOT / f"byte_{b:02d}"
    trace_files, meta_files = find_byte_files(byte_dir, b)

    meta_rows = 0
    for mf in meta_files:
        try:
            meta_rows += len(pd.read_csv(mf))
        except Exception:
            pass

    rows.append({
        "byte": b,
        "byte_dir": str(byte_dir),
        "trace_chunks": len(trace_files),
        "metadata_chunks": len(meta_files),
        "metadata_rows": meta_rows,
        "expected_rows": 30000,
        "looks_ok": len(trace_files) == len(meta_files) and meta_rows == 30000,
    })

verify_df = pd.DataFrame(rows)
verify_path = RESULTS_ROOT / "dataset_quick_verification_before_training.csv"
verify_df.to_csv(verify_path, index=False)
print("Saved quick verification:", verify_path)
display(verify_df)

if not verify_df["looks_ok"].all():
    print("WARNING: Some bytes may not have 30 chunks / 30,000 rows. Check before training all bytes.")


# CELL TYPE: markdown
## 5. Save Training Configuration

# CELL TYPE: code
training_config = {
    "project_root": str(PROJECT_ROOT),
    "dataset_root": str(DATASET_ROOT),
    "results_root": str(RESULTS_ROOT),
    "models_root": str(MODELS_ROOT),
    "bytes_to_train": BYTES_TO_TRAIN,
    "topk_values": TOPK_VALUES,
    "group_sizes": GROUP_SIZES,
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "use_augmentation": USE_AUGMENTATION,
    "aug_shift_max": AUG_SHIFT_MAX,
    "aug_noise_std": AUG_NOISE_STD,
    "aug_scale_std": AUG_SCALE_STD,
    "device": device,
    "run_id": RUN_ID,
}

config_path = RESULTS_ROOT / "training_config.json"
config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")
print("Saved config:", config_path)

# CELL TYPE: markdown
## 6. Data Loading Functions

# CELL TYPE: code
def get_byte_files(dataset_root: Path, target_byte: int):
    byte_dir = dataset_root / f"byte_{target_byte:02d}"

    trace_patterns = [
        f"byte{target_byte:02d}_chunk_*_traces.npz",
        f"ascon_byte{target_byte:02d}_chunk_*_traces.npz",
        "*_traces.npz",
    ]
    meta_patterns = [
        f"byte{target_byte:02d}_chunk_*_metadata.csv",
        f"ascon_byte{target_byte:02d}_chunk_*_metadata.csv",
        "*_metadata.csv",
    ]

    trace_files = []
    meta_files = []
    for pat in trace_patterns:
        trace_files = sorted(byte_dir.glob(pat))
        if trace_files:
            break
    for pat in meta_patterns:
        meta_files = sorted(byte_dir.glob(pat))
        if meta_files:
            break

    if not trace_files:
        raise FileNotFoundError(f"No trace npz files found for byte {target_byte:02d} in {byte_dir}")
    if not meta_files:
        raise FileNotFoundError(f"No metadata csv files found for byte {target_byte:02d} in {byte_dir}")

    if len(trace_files) != len(meta_files):
        raise ValueError(f"Chunk count mismatch for byte {target_byte:02d}: traces={len(trace_files)}, metadata={len(meta_files)}")

    return byte_dir, trace_files, meta_files


def load_byte_dataset(dataset_root: Path, target_byte: int):
    byte_dir, trace_files, meta_files = get_byte_files(dataset_root, target_byte)

    print(f"\nLoading ASCON byte {target_byte:02d}")
    print("Byte dir:", byte_dir)
    print("Trace chunks:", len(trace_files))
    print("Metadata chunks:", len(meta_files))

    X_list = []
    meta_list = []

    for tf, mf in zip(trace_files, meta_files):
        with np.load(tf) as npz:
            if "traces" in npz.files:
                traces = npz["traces"]
            else:
                traces = npz[npz.files[0]]

        meta = pd.read_csv(mf)

        if len(traces) != len(meta):
            raise ValueError(f"Row mismatch: {tf.name} has {len(traces)} traces but {mf.name} has {len(meta)} rows")

        X_list.append(traces.astype(np.float32, copy=False))
        meta_list.append(meta)

    X = np.vstack(X_list).astype(np.float32, copy=False)
    meta = pd.concat(meta_list, ignore_index=True)

    required_cols = ["target_byte", "key_hex", "key_byte_value", "key_byte_hex", "key_byte_hw"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing metadata columns for byte {target_byte:02d}: {missing}")

    y = meta["key_byte_value"].astype(np.int64).values

    # Basic label sanity check.
    bad = 0
    for i, row in meta.head(min(5000, len(meta))).iterrows():
        kh = str(row["key_hex"])
        tb = int(row["target_byte"])
        kv = int(row["key_byte_value"])
        if len(kh) >= (tb * 2 + 2):
            kv_from_key = int(kh[tb*2:tb*2+2], 16)
            if kv_from_key != kv:
                bad += 1
    if bad > 0:
        print(f"WARNING: Found {bad} key label mismatches in first checked rows for byte {target_byte:02d}")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class count:", len(np.unique(y)))
    print("Trace dtype:", X.dtype)

    return X, y, meta


def safe_stratify_labels(y, min_count_required=2):
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        return None
    if counts.min() < min_count_required:
        print("Warning: disabling stratify because at least one class has too few samples.")
        return None
    return y


def preprocess_and_split(X, y, seed=42):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Remove per-trace DC offset.
    X = X - X.mean(axis=1, keepdims=True)

    indices = np.arange(len(y))
    stratify_y = safe_stratify_labels(y, min_count_required=3)

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices,
        test_size=TEST_SIZE_TOTAL,
        random_state=seed,
        stratify=stratify_y,
    )

    stratify_temp = safe_stratify_labels(y_temp, min_count_required=2)

    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=VAL_SIZE_FROM_TEMP,
        random_state=seed,
        stratify=stratify_temp,
    )

    # Standardize per sample point using train statistics only.
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    X_train = ((X_train - mean) / std).astype(np.float32, copy=False)
    X_val = ((X_val - mean) / std).astype(np.float32, copy=False)
    X_test = ((X_test - mean) / std).astype(np.float32, copy=False)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test, mean, std

# CELL TYPE: markdown
## 7. Dataset Class

# CELL TYPE: code
class TraceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _augment_trace(self, x):
        if AUG_SHIFT_MAX and AUG_SHIFT_MAX > 0:
            shift = int(torch.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, (1,)).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

        if AUG_SCALE_STD and AUG_SCALE_STD > 0:
            scale = 1.0 + torch.randn(1).item() * AUG_SCALE_STD
            x = x * scale

        if AUG_NOISE_STD and AUG_NOISE_STD > 0:
            x = x + torch.randn_like(x) * AUG_NOISE_STD

        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = self._augment_trace(x)
        return x.unsqueeze(0), y


def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    train_ds = TraceDataset(X_train, y_train, augment=USE_AUGMENTATION)
    val_ds = TraceDataset(X_val, y_val, augment=False)
    test_ds = TraceDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

# CELL TYPE: markdown
## 8. Custom 1D ResCNN 256-Class Model

# CELL TYPE: code
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.05):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        out = self.act(out)
        return out


class CNN256ResCNN1D(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock1D(32, 32, kernel_size=7, stride=1, dropout=0.03),
            ResidualBlock1D(32, 32, kernel_size=7, stride=1, dropout=0.03),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(32, 64, kernel_size=7, stride=2, dropout=0.05),
            ResidualBlock1D(64, 64, kernel_size=7, stride=1, dropout=0.05),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=5, stride=2, dropout=0.08),
            ResidualBlock1D(128, 128, kernel_size=5, stride=1, dropout=0.08),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=0.10),
            ResidualBlock1D(256, 256, kernel_size=5, stride=1, dropout=0.10),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        x = torch.cat([avg, mx], dim=1)
        return self.head(x)


_tmp = CNN256ResCNN1D(num_classes=256)
print("Model parameters:", f"{sum(p.numel() for p in _tmp.parameters()):,}")
del _tmp

# CELL TYPE: markdown
## 9. Metrics

# CELL TYPE: code
def topk_accuracy_from_logits(logits, y, ks=TOPK_VALUES):
    results = {}
    max_k = max(ks)
    topk = logits.topk(max_k, dim=1).indices
    for k in ks:
        results[f"top{k}"] = topk[:, :k].eq(y.view(-1, 1)).any(dim=1).float().mean().item()
    return results


@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * len(yb)
        total_samples += len(yb)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)
    avg_loss = total_loss / max(total_samples, 1)
    topk = topk_accuracy_from_logits(all_logits, all_y, ks=TOPK_VALUES)
    return avg_loss, topk, all_logits, all_y


def grouped_topk_from_logits(logits, y, group_size=4, ks=TOPK_VALUES, seed=42):
    logits = logits.detach().cpu()
    y_np = y.detach().cpu().numpy()
    log_probs = F.log_softmax(logits, dim=1).numpy()
    rng = np.random.default_rng(seed)

    group_scores = []
    group_labels = []

    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        if len(idx) < group_size:
            continue
        rng.shuffle(idx)
        n_groups = len(idx) // group_size
        idx = idx[: n_groups * group_size]
        idx_groups = idx.reshape(n_groups, group_size)

        for g in idx_groups:
            score = log_probs[g].sum(axis=0)
            group_scores.append(score)
            group_labels.append(cls)

    result = {f"grouped_n{group_size}_groups": len(group_labels)}

    if len(group_scores) == 0:
        for k in ks:
            result[f"grouped_n{group_size}_top{k}"] = np.nan
        return result

    group_scores = torch.tensor(np.vstack(group_scores), dtype=torch.float32)
    group_labels = torch.tensor(np.array(group_labels), dtype=torch.long)
    max_k = max(ks)
    topk = group_scores.topk(max_k, dim=1).indices

    for k in ks:
        acc = topk[:, :k].eq(group_labels.view(-1, 1)).any(dim=1).float().mean().item()
        result[f"grouped_n{group_size}_top{k}"] = acc

    return result


def all_grouped_metrics(logits, y, group_sizes=GROUP_SIZES, seed=42):
    result = {}
    for n in group_sizes:
        result.update(grouped_topk_from_logits(logits, y, group_size=n, ks=TOPK_VALUES, seed=seed + n))
    return result


def make_top32_predictions_df(logits, y, idx_test):
    probs = F.softmax(logits, dim=1)
    top_probs, top_idx = probs.topk(32, dim=1)
    df = pd.DataFrame({
        "original_loaded_index": idx_test,
        "true_key_byte_value": y.numpy().astype(int),
        "true_key_byte_hex": [f"{int(v):02x}" for v in y.numpy()],
    })
    for rank in range(32):
        df[f"rank{rank+1}_value"] = top_idx[:, rank].numpy().astype(int)
        df[f"rank{rank+1}_hex"] = [f"{int(v):02x}" for v in top_idx[:, rank].numpy()]
        df[f"rank{rank+1}_prob"] = top_probs[:, rank].numpy()
    return df

# CELL TYPE: markdown
## 10. Train One Byte

# CELL TYPE: code
def train_one_byte(target_byte: int):
    byte_start = time.time()
    set_seed(SEED + target_byte)

    byte_result_dir = RESULTS_ROOT / f"byte_{target_byte:02d}"
    byte_model_dir = MODELS_ROOT / f"byte_{target_byte:02d}"
    byte_result_dir.mkdir(parents=True, exist_ok=True)
    byte_model_dir.mkdir(parents=True, exist_ok=True)

    summary_path = byte_result_dir / f"byte_{target_byte:02d}_summary.csv"
    if SKIP_COMPLETED_BYTES and summary_path.exists():
        print(f"Skipping byte {target_byte:02d}; existing summary found: {summary_path}")
        return pd.read_csv(summary_path).iloc[0].to_dict()

    X, y, meta = load_byte_dataset(DATASET_ROOT, target_byte)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        idx_train, idx_val, idx_test,
        mean, std,
    ) = preprocess_and_split(X, y, seed=SEED + target_byte)

    if SAVE_SPLIT_MAPPING:
        split_df = meta.copy()
        split_df["original_loaded_index"] = np.arange(len(split_df))
        split_df["split"] = "unused"
        split_df.loc[idx_train, "split"] = "train"
        split_df.loc[idx_val, "split"] = "validation"
        split_df.loc[idx_test, "split"] = "test"
        split_df.to_csv(byte_result_dir / f"byte_{target_byte:02d}_split_mapping.csv", index=False)

    train_loader, val_loader, test_loader = make_loaders(X_train, X_val, X_test, y_train, y_val, y_test)

    model = CNN256ResCNN1D(num_classes=256).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_top4 = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []
    best_model_path = byte_model_dir / f"byte_{target_byte:02d}_ascon_cnn256_rescnn_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        loader_iter = train_loader
        if TQDM_AVAILABLE:
            loader_iter = tqdm(train_loader, leave=False, desc=f"Byte {target_byte:02d} Epoch {epoch:02d}")

        for xb, yb in loader_iter:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_samples += len(yb)

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss, val_topk, _, _ = evaluate_model(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "target_byte": target_byte,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        }
        row.update({f"val_{k}": v for k, v in val_topk.items()})
        history.append(row)

        print(
            f"Byte {target_byte:02d} | Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e} | "
            f"Top1={val_topk['top1']:.4f} | Top2={val_topk['top2']:.4f} | "
            f"Top4={val_topk['top4']:.4f} | Top8={val_topk['top8']:.4f} | "
            f"Top16={val_topk['top16']:.4f} | Top32={val_topk['top32']:.4f}"
        )

        scheduler.step(val_topk["top4"])

        if val_topk["top4"] > best_val_top4:
            best_val_top4 = val_topk["top4"]
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "target_byte": target_byte,
                "algorithm": "ASCON-AEAD-128",
                "model_name": "CNN256ResCNN1D",
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_top4": best_val_top4,
                "topk_values": TOPK_VALUES,
                "mean": torch.tensor(mean, dtype=torch.float32),
                "std": torch.tensor(std, dtype=torch.float32),
                "config": training_config,
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best val Top4: {best_val_top4:.4f}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(byte_result_dir / f"byte_{target_byte:02d}_training_history.csv", index=False)

    # PyTorch 2.6+ defaults to weights_only=True. This checkpoint is generated locally by this notebook.
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_topk, test_logits, test_y = evaluate_model(model, test_loader, criterion)
    grouped_metrics = all_grouped_metrics(test_logits, test_y, group_sizes=GROUP_SIZES, seed=SEED + target_byte)

    if SAVE_TOP32_PREDICTIONS:
        pred_df = make_top32_predictions_df(test_logits, test_y, idx_test)
        pred_df.to_csv(byte_result_dir / f"byte_{target_byte:02d}_test_predictions_top32.csv", index=False)

    elapsed_min = (time.time() - byte_start) / 60.0

    result = {
        "target_byte": target_byte,
        "algorithm": "ASCON-AEAD-128",
        "model": "CNN256ResCNN1D",
        "best_epoch": best_epoch,
        "best_val_top4": best_val_top4,
        "test_loss": test_loss,
        "elapsed_minutes": elapsed_min,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "class_count_total": len(np.unique(y)),
        "trace_len": int(X.shape[1]),
    }
    result.update(test_topk)
    result.update(grouped_metrics)

    pd.DataFrame([result]).to_csv(summary_path, index=False)

    print(f"\nFinished ASCON byte {target_byte:02d}")
    print("Best epoch:", best_epoch)
    print("Test Top-k:", test_topk)
    print("Grouped Top4:", {k: v for k, v in grouped_metrics.items() if "top4" in k})
    print("Elapsed minutes:", f"{elapsed_min:.2f}")
    print("Saved model:", best_model_path)
    print("Saved summary:", summary_path)

    # Clean memory before next byte.
    del X, y, meta, X_train, X_val, X_test, y_train, y_val, y_test
    del train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# CELL TYPE: markdown
## 11. Train Selected Bytes

# CELL TYPE: code
all_results = []
start_all = time.time()

for b in BYTES_TO_TRAIN:
    result = train_one_byte(b)
    all_results.append(result)

summary_df = pd.DataFrame(all_results).sort_values("target_byte").reset_index(drop=True)
summary_path = RESULTS_ROOT / "summary_results_ascon_cnn256_rescnn.csv"
summary_df.to_csv(summary_path, index=False)

elapsed_all_hours = (time.time() - start_all) / 3600.0
print("\nAll requested ASCON bytes finished.")
print("Elapsed hours:", f"{elapsed_all_hours:.2f}")
print("Saved all-byte summary:", summary_path)

display(summary_df)

# CELL TYPE: markdown
## 12. Candidate-Space Reduction Summary

# CELL TYPE: code
def candidate_space_power(k, n_bytes=16):
    return n_bytes * math.log2(k)


def build_single_trace_candidate_summary(summary_df):
    rows = []
    for k in TOPK_VALUES:
        col = f"top{k}"
        if col not in summary_df.columns:
            continue
        values = summary_df[col].astype(float).values
        weakest_idx = int(np.nanargmin(values))
        weakest_byte = int(summary_df.loc[weakest_idx, "target_byte"])
        weakest_value = float(values[weakest_idx])
        avg_value = float(np.nanmean(values))
        full_key_inclusion_estimate = float(np.nanprod(values))
        log2_space = candidate_space_power(k, n_bytes=16)
        rows.append({
            "setting": f"single_trace_top{k}",
            "k": k,
            "average_byte_accuracy": avg_value,
            "weakest_byte": weakest_byte,
            "weakest_byte_accuracy": weakest_value,
            "estimated_full_key_inclusion_probability": full_key_inclusion_estimate,
            "candidate_space_expression": f"{k}^16",
            "candidate_space_log2": log2_space,
            "candidate_space_as_power_of_2": f"2^{log2_space:.0f}" if abs(log2_space - round(log2_space)) < 1e-9 else f"2^{log2_space:.2f}",
        })
    return pd.DataFrame(rows)


def build_grouped_candidate_summary(summary_df):
    rows = []
    for n in GROUP_SIZES:
        for k in TOPK_VALUES:
            col = f"grouped_n{n}_top{k}"
            if col not in summary_df.columns:
                continue
            values = summary_df[col].astype(float).values
            if np.all(np.isnan(values)):
                continue
            weakest_idx = int(np.nanargmin(values))
            weakest_byte = int(summary_df.loc[weakest_idx, "target_byte"])
            weakest_value = float(values[weakest_idx])
            avg_value = float(np.nanmean(values))
            full_key_inclusion_estimate = float(np.nanprod(values))
            log2_space = candidate_space_power(k, n_bytes=16)
            rows.append({
                "setting": f"grouped_n{n}_top{k}",
                "group_size": n,
                "k": k,
                "average_byte_accuracy": avg_value,
                "weakest_byte": weakest_byte,
                "weakest_byte_accuracy": weakest_value,
                "estimated_full_key_inclusion_probability": full_key_inclusion_estimate,
                "candidate_space_expression": f"{k}^16",
                "candidate_space_log2": log2_space,
                "candidate_space_as_power_of_2": f"2^{log2_space:.0f}" if abs(log2_space - round(log2_space)) < 1e-9 else f"2^{log2_space:.2f}",
            })
    return pd.DataFrame(rows)

single_candidate_df = build_single_trace_candidate_summary(summary_df)
grouped_candidate_df = build_grouped_candidate_summary(summary_df)

single_path = RESULTS_ROOT / "candidate_space_summary_single_trace_ascon.csv"
grouped_path = RESULTS_ROOT / "candidate_space_summary_grouped_ascon.csv"

single_candidate_df.to_csv(single_path, index=False)
grouped_candidate_df.to_csv(grouped_path, index=False)

print("Saved single-trace candidate summary:", single_path)
print("Saved grouped candidate summary:", grouped_path)

display(single_candidate_df)
if not grouped_candidate_df.empty:
    display(grouped_candidate_df[grouped_candidate_df["setting"].str.contains("top4")].sort_values("group_size"))
else:
    print("No grouped candidate summary available.")

# CELL TYPE: markdown
## 13. Quick Result View

# CELL TYPE: code
cols = [
    "target_byte", "best_epoch", "best_val_top4",
    "top1", "top2", "top4", "top8", "top16", "top32",
    "grouped_n2_top4", "grouped_n4_top4", "grouped_n8_top4", "grouped_n12_top4", "grouped_n16_top4",
]
cols = [c for c in cols if c in summary_df.columns]
summary_df[cols]

# CELL TYPE: markdown
## Notes

For ASCON, this notebook follows the agreed folder rule:

```text
03_Data   = traces only
04_Results = training CSVs, summaries, plots, predictions, split mappings
05_Models  = .pt model checkpoints only
```

If CUDA runs out of memory, stop the notebook, change `BATCH_SIZE = 64`, and rerun from the beginning.
