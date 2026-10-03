"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_split(processed_dir: str | None = None):
    """Nạp train và eval từ file .npz."""
    if processed_dir is None:
        processed_dir = str(REPO_ROOT / "data" / "processed")
    train_path = str(Path(processed_dir) / "train.npz")
    eval_path = str(Path(processed_dir) / "eval.npz")

    with np.load(train_path) as train_npz:
        X_train_full = train_npz["X"]
        y_train_full = train_npz["y"]
        assert X_train_full.dtype == np.float32, X_train_full.dtype
        assert y_train_full.dtype == np.int64, y_train_full.dtype

    with np.load(eval_path) as eval_npz:
        X_eval = eval_npz["X"]
        y_eval = eval_npz["y"]
        eval_row_id = eval_npz["row_id"]
        assert X_eval.dtype == np.float32, X_eval.dtype
        assert y_eval.dtype == np.int64, y_eval.dtype
        assert eval_row_id.dtype == np.int64, eval_row_id.dtype

    assert X_train_full.shape[1] == 54, X_train_full.shape
    assert X_eval.shape[1] == 54, X_eval.shape
    assert y_train_full.shape[0] == X_train_full.shape[0], (y_train_full.shape, X_train_full.shape)
    assert y_eval.shape[0] == X_eval.shape[0], (y_eval.shape, X_eval.shape)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation từ train theo stratify."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean/std của 10 cột liên tục trên train."""
    X_num = X_tr[:, :N_NUMERIC]
    mean = X_num.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_num.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Chuẩn hoá 10 cột đầu, giữ nguyên 44 cột nhị phân."""
    X = X.copy()
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str | None = None) -> dict:
    """Gộp dữ liệu và đưa lên device."""
    X_tr_full, y_tr_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_tr_full, y_tr_full, val_fraction=val_fraction, seed=seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    device = torch.device(device)
    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.int64, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.int64, device=device)
    X_eval = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval = torch.tensor(y_eval, dtype=torch.int64, device=device)

    majority = int(np.bincount(y_tr_full).argmax())
    val_acc_majority = float((y_val.cpu().numpy() == majority).mean())
    print(f"train shape={X_tr.shape}, val shape={X_val.shape}, eval shape={X_eval.shape}")
    print(f"majority-class accuracy on val ~ {val_acc_majority:.4f}")
    print(f"mean numeric ~ {X_tr[:, :N_NUMERIC].mean(dim=0).cpu().numpy()[:3]}")
    print(f"std numeric ~ {X_tr[:, :N_NUMERIC].std(dim=0, unbiased=False).cpu().numpy()[:3]}")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "device": device,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb)."""
    N = X.shape[0]
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
