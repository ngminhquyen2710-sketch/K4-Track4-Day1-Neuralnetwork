"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ một thí nghiệm thành 3 ô."""
    hist = result["history"]
    cfg = result["cfg"]
    epochs = hist["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="train loss", marker="o", ms=3)
    ax.plot(epochs, hist["val_loss"], label="val loss", marker="s", ms=3)
    if "best_epoch" in result["summary"]:
        ax.axvline(result["summary"]["best_epoch"], color="gray", linestyle="--", alpha=0.7, label="best epoch")
    ax.set_title(f"{cfg['exp_id']} — loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="val acc", marker="o", ms=3)
    ax.plot(epochs, hist["val_macro_f1"], label="val macro-F1", marker="s", ms=3)
    if "best_epoch" in result["summary"]:
        ax.axvline(result["summary"]["best_epoch"], color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"{cfg['exp_id']} — accuracy / macro-F1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="grad norm", marker="^", ms=3, color="tab:green")
    ax.set_title(f"{cfg['exp_id']} — grad norm")
    ax.set_xlabel("epoch")
    ax.set_ylabel("||grad||")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle(f"exp_id={cfg['exp_id']} | opt={cfg['optimizer']} | lr={cfg['lr']} | batch={cfg['batch']} | hidden={cfg['hidden']} | dropout={cfg['dropout']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ các đường so sánh của nhiều thí nghiệm trên cùng một trục."""
    plt.figure(figsize=(10, 4.5))
    for res in results:
        hist = res["history"]
        if metric not in hist:
            continue
        plt.plot(hist["epoch"], hist[metric], label=res["cfg"]["exp_id"], marker="o", ms=2)
    plt.title(title or metric)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
