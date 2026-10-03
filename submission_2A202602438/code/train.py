"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=0.05,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64."""
    model.eval()
    all_preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_preds.append(preds)
    if not all_preds:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(all_preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Đánh giá loss/acc/macro-F1 trên tập cho trước."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    cm = np.zeros((7, 7), dtype=np.int64)

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item() * yb.numel())
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)

    loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    macro_f1 = macro_f1_from_confusion(cm)
    return {"loss": loss, "acc": acc, "macro_f1": macro_f1, "cm": cm}


def compute_loss(logits, y, loss_name: str):
    """Compute cross-entropy or MSE."""
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, y_onehot)
    raise ValueError(f"Unsupported loss name '{loss_name}'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình."""
    set_seed(int(cfg["seed"]))
    device = data["device"]
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"])
    assert hidden in EXPECTED_PARAMS, f"hidden={hidden} chưa có trong EXPECTED_PARAMS"
    assert count_params(model) == EXPECTED_PARAMS[hidden], (count_params(model), EXPECTED_PARAMS[hidden])
    model.to(device)

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        momentum=float(cfg["momentum"]),
    )

    scaler = None
    if cfg["precision"] == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler(device=device.type, enabled=True)

    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"], batch_size=max(256, int(cfg["batch"])))
    step0_loss = step0_eval["loss"]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False
    peak_mem_MB = 0.0

    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(int(cfg["seed"]))

    train_subset_size = min(50_000, data["X_tr"].shape[0])
    train_subset_X = data["X_tr"][:train_subset_size]
    train_subset_y = data["y_tr"][:train_subset_size]

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_grad_norm_values = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            if cfg["precision"] == "fp32":
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                optimizer.step()
            else:
                if device.type == "cuda" and cfg["precision"] == "fp16":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(xb)
                        loss = compute_loss(logits, yb, cfg["loss"])
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                elif device.type == "cpu" and cfg["precision"] == "bf16":
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        logits = model(xb)
                        loss = compute_loss(logits, yb, cfg["loss"])
                    loss.backward()
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                    optimizer.step()
                else:
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                    loss.backward()
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                    optimizer.step()

            if not torch.isfinite(loss):
                diverged = True
                break
            epoch_grad_norm_values.append(float(gn))

        if diverged:
            break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mem_MB = max(peak_mem_MB, torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            torch.cuda.reset_peak_memory_stats(device)

        model.eval()
        train_metrics = evaluate(model, train_subset_X, train_subset_y, loss_name=cfg["loss"], batch_size=max(2048, int(cfg["batch"])))
        val_metrics = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"], batch_size=max(2048, int(cfg["batch"])))

        epoch_time = time.perf_counter() - epoch_start
        avg_grad_norm = float(np.mean(epoch_grad_norm_values)) if epoch_grad_norm_values else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(avg_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = max(len(history["epoch"]), 1)

    final_epoch = history["epoch"][-1] if history["epoch"] else 1
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else step0_loss
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else step0_loss
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    final_val_macro_f1 = history["val_macro_f1"][-1] if history["val_macro_f1"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch if best_epoch > 0 else final_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(final_val_acc),
        "val_macro_f1": float(final_val_macro_f1),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    result = {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}
    return result


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file CSV predictions_eval.csv."""
    row_id = np.asarray(row_id)
    preds = np.asarray(preds, dtype=np.int64)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "pred"])
        for rid, p in zip(row_id, preds):
            writer.writerow([int(rid), int(p)])


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng best_state dự đoán trên eval."""
    device = data["device"]
    model = MLP(tuple(cfg["hidden"]), dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()
    preds = predict(model, data["X_eval"], batch_size=max(2048, int(cfg["batch"])))
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
