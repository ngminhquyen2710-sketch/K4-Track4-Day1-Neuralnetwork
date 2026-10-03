# Báo cáo Lab Day 1 — Nguyễn Minh Quyền — 2A202602438

> Dữ liệu và thí nghiệm đã chạy trên môi trường local với CPU. Mục tiêu là xây dựng MLP theo quy định, đánh giá trên validation, và chọn cấu hình cuối cùng bằng val trước khi chấm eval.

## 1. Thiết lập

- Môi trường: Python 3.14, PyTorch 2.14.1, scikit-learn 1.9.1, NumPy 2.5.3, Matplotlib 3.11.2.
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203, validation tách từ train theo stratify 20% (seed 42), còn lại 371 847 train / 92 962 val.
- Model: `M-base` = 54 -> 256 -> 128 -> 7, ReLU, bias, 47 879 tham số, không BatchNorm, không residual, logits `(B,7)`.
- Baseline: loss = cross-entropy, optimizer = SGD + momentum 0.9, lr = 0.05, batch = 512, epochs = 20, init = He, dropout = 0, clip_norm = None, precision = fp32.
- Mốc thấp nhất: accuracy của chiến lược “đoán luôn lớp 1” trên val = 0.4876.
- Chủ đề đã thử: loss, optimizer, dropout, clipping.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / `(B, 7)` |
| Loss bước 0 trên val | 2.2691, gần `ln 7 = 1.946` |
| Quá khớp 20 mẫu: loss cuối | gần 0 |
| Gradient có chảy | có |
| Baseline: val accuracy | 0.8976 |
| Baseline: val macro-F1 | 0.8371 |
| Thí nghiệm optimizer tốt nhất | `opt-adam-1e3`: val macro-F1 = 0.8475 |
| Thí nghiệm dropout | `dropout-q01`: val macro-F1 = 0.8248 |
| Thí nghiệm clipping | `clip-lr01`: val macro-F1 = 0.8448 |

Vì chỉ chạy 1 seed cho mỗi mô hình, độ nhiễu seed chưa được đo đầy đủ. Đây là hạn chế của báo cáo hiện tại.

## 3. Kết quả theo chủ đề

### 3.1 Baseline

- Dự đoán: baseline nên vượt mốc “đoán đa số” và thấy val loss giảm rõ trong 20 epoch.
- Kết quả: `exp_id = base-s1`, ảnh `figures/base-s1.png`.
- Thống kê chính: `best_val_loss = 0.25647`, `val_acc = 0.89759`, `val_macro_f1 = 0.83710` ở epoch 20.
- Giải thích: He init và SGD+momentum cho gradient ổn định, số bước cập nhật đủ để mô hình học tốt.

### 3.2 Bộ tối ưu hoá

- Dự đoán: Adam với lr = 1e-3 có thể ổn định hơn SGD ở cùng số epoch, vì Adam điều chỉnh tốc độ học theo moment bậc 1/2.
- Kết quả: `exp_id = opt-adam-1e3`, ảnh `figures/opt-adam-1e3.png`, `compare_optimizer.png`.
- Số liệu: val macro-F1 = 0.8475, cao hơn baseline 0.8371, best epoch = 20.
- Giải thích: Adam có thể hội tụ nhanh hơn trên dữ liệu này và không cần lr cao như SGD; mặt khác, trên bài toán này mất cân bằng lớp, Adam cải thiện tốt hơn phần F1 macro.

### 3.3 Dropout

- Dự đoán: với q = 0.1, nếu model chưa quá khớp thì dropout có thể giúp giảm overfit nhưng nếu q quá lớn sẽ làm mất thông tin.
- Kết quả: `exp_id = dropout-q01`, ảnh `figures/dropout-q01.png`, `compare_dropout.png`.
- Số liệu: val macro-F1 = 0.8248, thấp hơn baseline 0.8371.
- Giải thích: dropout ở q=0.1 trên dữ liệu này làm giảm mạnh năng lực học mà không mang lại cải thiện đáng kể; vì vậy không được chọn làm cấu hình cuối cùng.

### 3.4 Gradient clipping

- Dự đoán: ở lr = 0.1, clipping sẽ giúp nếu gradient bắt đầu tăng quá mức.
- Kết quả: `exp_id = clip-lr01`, ảnh `figures/clip-lr01.png`, `compare_clipping.png`.
- Số liệu: val macro-F1 = 0.8448, gần vượt baseline nhưng không bằng Adam.
- Giải thích: clipping có tác dụng ổn định nhưng ở bài toán này hiệu quả không lớn hơn việc đổi optimizer; nó giúp khi lr cao nhưng không phải là khắc phục chính cho mô hình này.

## 4. Đánh giá cuối trên tập eval

Chọn cấu hình bằng val: cấu hình cuối cùng là `opt-adam-1e3` vì đạt val macro-F1 cao nhất. Tệp dự đoán được tạo theo `predictions_eval.csv`, và chạy `scripts/evaluate.py --pred predictions_eval.csv --out eval_result.json` cho kết quả:

| Cấu hình | val macro-F1 | eval macro-F1 | eval accuracy |
|---|---|---|---|
| Baseline (`base-s1`) | 0.8371 | 0.8350 | 0.8958 |
| Cấu hình cuối cùng (`opt-adam-1e3`) | 0.8475 | 0.8487 | 0.9042 |

Val và eval gần nhau: chênh lệch dưới 0.01, nên validation là ước lượng đáng tin cậy.

### 4.1 Phân tích lỗi theo lớp

Từ `eval_result.json`, lớp khó nhất là lớp 4 với F1 ≈ 0.7520, và lớp 4 bị nhầm nhiều sang lớp 1. Lớp 3 có số lượng sample cực ít nên F1 thấp hơn dù precision cao (0.9027) nhưng recall chỉ 0.6758. Đa số lỗi tập trung vào lớp 1 vì đây là lớp đa số và hơi chồng lấn với các lớp khác khi đặc trưng địa hình tương đồng.

## 5. Trả lời các câu hỏi dẫn dắt

1. Bộ tối ưu nào thắng khi mỗi bộ được chỉnh lr công bằng? Trong thí nghiệm này, Adam đạt val macro-F1 cao nhất (0.8475), vượt SGD+momentum baseline (0.8371).
2. Dropout có giúp không khi mô hình chưa quá khớp? Trong trường hợp này, q=0.1 không giúp, vì val macro-F1 thấp hơn baseline. Dropout chỉ hợp khi mô hình bắt đầu overfitting rõ ràng.
3. Gradient clipping giải quyết vấn đề gì? Nó giúp ổn định gradient ở lr cao, nhưng ở đây không phải là yếu tố quyết định như optimizer.
4. Mixed precision có làm huấn luyện nhanh hơn không? Chưa thử trên thí nghiệm này, đây là phần cần mở rộng tiếp theo.
5. Vì sao khởi tạo toàn số 0 hỏng? Vì tất cả tham số đều giống nhau, các đơn vị không thể phá vỡ đối xứng. He init giúp duy trì variance phù hợp với ReLU.
6. Quay lại câu hỏi của bài học: “có loss không giảm sau 2000 bước.” Ba kiểm tra đầu tiên là: loss bước 0, quá khớp 20 mẫu, và gradient có chảy không. Nếu lần lượt đều ổn, có khả năng vấn đề nằm ở learning rate / optimizer / dữ liệu, không phải ở bug code.

## 6. Hạn chế

- Chỉ chạy 1 seed cho mỗi cấu hình, nên chưa có ước lượng nhiễu robust.
- Chưa thử mixed precision, khởi tạo khác hoặc các kiến trúc rộng/deep.
- Môi trường chạy là CPU, nên thời gian huấn luyện dài hơn GPU.

## 7. Phụ lục

- `base-s1.json`, `opt-adam-1e3.json`, `dropout-q01.json`, `clip-lr01.json` trong [results](results/)
- `base-s1.png`, `opt-adam-1e3.png`, `dropout-q01.png`, `clip-lr01.png`, `compare_optimizer.png`, `compare_dropout.png`, `compare_clipping.png` trong [figures](figures/)
- `predictions_eval.csv`
- `eval_result.json`
