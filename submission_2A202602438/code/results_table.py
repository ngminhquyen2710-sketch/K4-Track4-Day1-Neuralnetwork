"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result thành JSON trong thư mục <results_dir>."""
    exp_id = result["cfg"]["exp_id"]
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_id}.json"
    payload = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir và trả về list sắp xếp theo exp_id."""
    out_dir = Path(results_dir)
    if not out_dir.exists():
        return []
    rows = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda x: x["cfg"]["exp_id"])
    return rows


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến kết quả thành dòng bảng Experiments."""
    cfg = result["cfg"]
    summary = result["summary"]
    row = {
        "exp_id": cfg["exp_id"],
        "group": cfg["group"],
        "description": cfg["description"],
        "loss": cfg["loss"],
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "hidden": str(cfg["hidden"]),
        "dropout": cfg["dropout"],
        "clip_norm": cfg["clip_norm"],
        "precision": cfg["precision"],
        "init": cfg["init"],
        "seed": cfg["seed"],
        "step0_loss": summary["step0_loss"],
        "best_val_loss": summary["best_val_loss"],
        "best_epoch": summary["best_epoch"],
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss": summary["final_val_loss"],
        "val_acc": summary["val_acc"],
        "val_macro_f1": summary["val_macro_f1"],
        "time_per_epoch_s": summary["time_per_epoch_s"],
        "peak_mem_MB": summary["peak_mem_MB"],
        "diverged": summary["diverged"],
        "eval_acc": eval_scores["accuracy"] if eval_scores else None,
        "eval_macro_f1": eval_scores["macro_f1"] if eval_scores else None,
        "figure_file": f"figures/{cfg['exp_id']}.png",
        "notes": notes,
    }
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền rows lên sheet Experiments của file template Excel."""
    wb = load_workbook(template_path)
    ws = wb["Experiments"]
    headers = [cell.value for cell in ws[1]]
    for row_idx, row in enumerate(rows, start=2):
        for col_idx, header in enumerate(headers, start=1):
            if header in row:
                ws.cell(row=row_idx, column=col_idx, value=row[header])
    wb.save(out_path)
