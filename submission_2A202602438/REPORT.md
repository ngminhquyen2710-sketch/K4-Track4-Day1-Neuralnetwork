# Báo cáo Lab Day 1 — Nguyễn Minh Quyền — 2A202602438

> Dữ liệu và thí nghiệm đã chạy trên môi trường local với CPU. Mục tiêu là xây dựng MLP theo quy định, đánh giá trên validation, và chọn cấu hình cuối cùng bằng val trước khi chấm eval.

## 1. Thiết lập

- Môi trường: Python 3.14, PyTorch 2.14.1, scikit-learn 1.9.1, NumPy 2.5.3, Matplotlib 3.11.2.
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203, validation tách từ train theo stratify 20% (seed 42), còn lại 371 847 train / 92 962 val.
- Model: `M-base` = 54 -> 256 -> 128 -> 7, ReLU, bias, 47 879 tham số, không BatchNorm, không residual, logits `(B,7)`.
- Baseline: loss = cross-entropy, optimizer = SGD + momentum 0.9, lr = 0.05, batch = 512, epochs = 20, init = He, dropout = 0, clip_norm = None, precision = fp32.
- Mốc thấp nhất: accuracy của chiến lược “đoán luôn lớp 1” trên val = 0.4876.
- Chủ đề đã thử: loss, optimizer, hparam, dropout, clipping, amp, init. Tổng cộng 7 chủ đề, và tất cả đều có ít nhất 1 thí nghiệm với `val_macro_f1`.

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
| Thí nghiệm hparam tốt nhất | `hparam-adamw`: val macro-F1 = 0.8486 |
| Thí nghiệm dropout | `dropout-q01`: val macro-F1 = 0.8248 |
| Thí nghiệm clipping | `clip-lr01`: val macro-F1 = 0.8448 |
| Thí nghiệm amp | `amp-bf16`: val macro-F1 = 0.8466 |
| Thí nghiệm init | `init-xavier`: val macro-F1 = 0.8462 |
| Thí nghiệm loss | `loss-mse`: val macro-F1 = 0.8339 |

So với dữ liệu tạo ra bằng một seed duy nhất, độ nhiễu seed chưa được đo đầy đủ. Đây là hạn chế của báo cáo, nhưng tất cả các chủ đề đã có ít nhất 1 số liệu lưu trên validation.

## 3. Kết quả theo chủ đề

### 3.1 Baseline

- Dự đoán: baseline nên vượt mốc “đoán đa số” và thấy val loss giảm rõ trong 20 epoch.
- Kết quả: `exp_id = base-s1`, ảnh `figures/base-s1.png`.
- Thống kê chính: `best_val_loss = 0.25647`, `val_acc = 0.89759`, `val_macro_f1 = 0.83710` ở epoch 20.
- Giải thích: He init và SGD+momentum cho gradient ổn định, số bước cập nhật đủ để mô hình học tốt.

### 3.2 Loss

- Dự đoán: cross-entropy phù hợp hơn MSE cho phân loại đa lớp, vì MSE trên one-hot không tối ưu hóa trực tiếp khoảng cách lợi ích của logit.
- Kết quả: `exp_id = loss-mse`, ảnh `figures/loss-mse.png`.
- Số liệu: `val_macro_f1 = 0.8339`, thấp hơn baseline `0.8371` nhưng vẫn rõ ràng vượt mốc “đoán đa số”.
- Giải thích: MSE xây dựng mục tiêu quá cứng và ít nhạy với mỗi lớp, nên khi dữ liệu mất cân bằng và nhãn có độ tương đồng, F1 macro bị giảm.

### 3.3 Bộ tối ưu hoá

- Dự đoán: Adam với lr = 1e-3 có thể ổn định hơn SGD ở cùng số epoch.
- Kết quả: `exp_id = opt-adam-1e3`, ảnh `figures/opt-adam-1e3.png`.
- Số liệu: `val_macro_f1 = 0.8475`, cao hơn baseline 0.8371.
- Giải thích: Adam điều chỉnh tốc độ học theo moment bậc 1 và 2, giúp hội tụ nhanh hơn và ổn định hơn trên tập dữ liệu mất cân bằng này.

### 3.4 Hyper-parameter (hparam)

- Dự đoán: điều chỉnh giảm trọng số bằng `AdamW` và `weight_decay=1e-4` sẽ cải thiện generalisation nhẹ so với Adam thuần.
- Kết quả: `exp_id = hparam-adamw`, ảnh `figures/hparam-adamw.png`.
- Số liệu: `val_macro_f1 = 0.8486`, cao nhất trong các lần chạy hiện tại. `best_epoch = 19`.
- Giải thích: `AdamW` tách weight decay khỏi grad, giúp tránh quá khớp nhẹ và cải thiện sự ổn định ở vùng cập nhật cuối cùng.

### 3.5 Dropout

- Dự đoán: với q = 0.1, nếu model chưa quá khớp thì dropout có thể giúp nhưng không phải lúc nào cũng hiệu quả.
- Kết quả: `exp_id = dropout-q01`, ảnh `figures/dropout-q01.png`.
- Số liệu: `val_macro_f1 = 0.8248`, thấp hơn baseline `0.8371`.
- Giải thích: q=0.1 làm giảm khả năng học trên bài toán này và không cải thiện F1 macro, vì dữ liệu phân loại đa lớp vốn đã khó.

### 3.6 Gradient clipping

- Dự đoán: ở lr = 0.1, clipping sẽ giúp nếu gradient bắt đầu tăng quá mức.
- Kết quả: `exp_id = clip-lr01`, ảnh `figures/clip-lr01.png`.
- Số liệu: `val_macro_f1 = 0.8448`, gần vượt baseline nhưng không bằng Adam / AdamW.
- Giải thích: clipping ổn định gradient nhưng không thay đổi bản chất của optimizer như Adam; nó là công cụ an toàn hơn, không phải giải pháp tối ưu chính.

### 3.7 Mixed precision (AMP)

- Dự đoán: bf16 trên CPU có thể giữ độ ổn định trong khi giảm chi phí tính toán, nhưng không nhất thiết cải thiện tương quan F1.
- Kết quả: `exp_id = amp-bf16`, ảnh `figures/amp-bf16.png`.
- Số liệu: `val_macro_f1 = 0.8466`, thấp hơn hparam AdamW nhưng cao hơn baseline và rất gần Adam.
- Giải thích: trên CPU, bf16 chắc chắn vẫn có giá trị cho tốc độ nhưng lợi ích trên métrik F1 không nổi bật hơn nhiều so với fp32.

### 3.8 Khởi tạo tham số (init)

- Dự đoán: Xavier init có thể hiệu quả cho tầng đầu ra, nhưng ReLU vẫn thích `He` init hơn trên MLP này.
- Kết quả: `exp_id = init-xavier`, ảnh `figures/init-xavier.png`.
- Số liệu: `val_macro_f1 = 0.8462`, gần bằng Adam nhưng không vượt `hparam-adamw`.
- Giải thích: He init phù hợp với ReLU vì duy trì độ biến thiên thích hợp ở tầng đầu vào, giúp thông tin truyền tốt qua các layer ẩn.

## 4. Đánh giá cuối trên tập eval

Chọn cấu hình bằng validation: cấu hình cuối cùng là `hparam-adamw` vì đạt `val_macro_f1` cao nhất hiện tại. Tệp dự đoán tạo từ `predictions_eval.csv`, và chạy `scripts/evaluate.py --pred predictions_eval.csv --out eval_result.json` cho kết quả:

| Cấu hình | val macro-F1 | eval macro-F1 | eval accuracy |
|---|---|---|---|
| Baseline (`base-s1`) | 0.8371 | 0.8350 | 0.8958 |
| `opt-adam-1e3` | 0.8475 | 0.8487 | 0.9042 |
| `hparam-adamw` | 0.8486 | 0.8531 | 0.9026 |

Val và eval gần nhau: chênh lệch dưới ~0.01, nên validation là ước lượng đáng tin cậy. Cấu hình cuối cùng được chọn là `hparam-adamw`.

### 4.1 Phân tích lỗi theo lớp

Từ `eval_result.json`, lớp khó nhất là lớp 4 với F1 ≈ 0.7513; nhiều lỗi của lớp 4 bị nhầm sang lớp 1. Lớp 3 có số lượng mẫu rất ít nên F1 thấp hơn dù precision khá cao. Đa số lỗi còn lại tập trung giữa lớp 1 và các lớp còn lại do đặc trưng địa hình tương đồng, đặc biệt ở cấu trúc rừng thường chồng lấn về yếu tố địa hình.

## 5. Trả lời các câu hỏi dẫn dắt

1. Bộ tối ưu nào thắng khi mỗi bộ được chỉnh lr công bằng? Trong các thử nghiệm, `AdamW` với `weight_decay=1e-4` thắng nhẹ, đạt `val_macro_f1 = 0.8486`.
2. Dropout có giúp không khi mô hình chưa quá khớp? Trong trường hợp này, q=0.1 không giúp: `val_macro_f1` thấp hơn baseline.
3. Gradient clipping giải quyết vấn đề gì? Nó giúp ổn định gradient ở lr cao, nhưng không phải là cột mốc cho hiệu suất tốt nhất.
4. Mixed precision có làm huấn luyện nhanh hơn không? Dù chưa được đánh giá sâu trên GPU, thử nghiệm `bf16` trên CPU cho thấy tính ổn định và hiệu suất tương tự `fp32`, không tăng F1 đáng kể.
5. Vì sao khởi tạo toàn số 0 hỏng? Vì tất cả tham số đều giống nhau, các đơn vị không thể phá vỡ đối xứng. `He` init giúp duy trì variance phù hợp với ReLU.
6. Quay lại câu hỏi của bài học: “có loss không giảm sau 2000 bước”. Ba kiểm tra đầu tiên là loss bước 0, giữ `train_loss` / `val_loss` giảm hay không, và tính gradient có chảy không. Nếu thứ nhất/ba đều ổn, vấn đề rất có thể nằm ở learning rate, optimizer hoặc dữ liệu chứ không phải bug code.

## 6. Hạn chế

- Chỉ chạy 1 seed cho mỗi cấu hình, nên chưa có ước lượng nhiễu robust.
- Môi trường chạy là CPU, nên thời gian huấn luyện lâu hơn so với GPU.
- Mặc dù đã đủ 7 chủ đề, các thí nghiệm chưa đi sâu ở nhiều seed và nhiều cấu hình hyper-parameter.

## 7. Phụ lục

- `base-s1.json`, `opt-adam-1e3.json`, `hparam-adamw.json`, `dropout-q01.json`, `clip-lr01.json`, `amp-bf16.json`, `loss-mse.json`, `init-xavier.json` trong [results](results/)
- `base-s1.png`, `opt-adam-1e3.png`, `hparam-adamw.png`, `dropout-q01.png`, `clip-lr01.png`, `amp-bf16.png`, `loss-mse.png`, `init-xavier.png` trong [figures](figures/)
- `predictions_eval.csv`
- `eval_result.json`
