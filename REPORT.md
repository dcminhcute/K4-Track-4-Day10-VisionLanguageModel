Họ tên: Đoàn Quang Minh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Lưu ý kỹ thuật: file `e0.json` và `e2.json` bị **đảo nhãn `name`** khi ghi (`e0.json` ghi `"name":"E2"`, `e2.json` ghi `"name":"E0"`). Số liệu bên trong vẫn đúng từng experiment (xác nhận qua `trainable_params`: `e0.json` không có `meta` → E0; `e1.json` = 33,030,144; `e2.json` = 142,135,296; `e3.json` = 148,426,752). Em không sửa file JSON để giữ nguyên evidence nhưng đã xác minh mapping bằng `meta` trước khi viết bảng.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.058 | [0.040, 0.078] | 1.367 | 0.102 | 0.007 | 0.135 |
| E1 | LoRA LLM | 33,030,144 | 9.674 | 48.854 | 0.276 | [0.239, 0.318] | 1.254 | 0.149 | 0.211 | 0.549 |
| E2 | + projector | 142,135,296 | 11.377 | 50.087 | 0.340 | [0.297, 0.381] | 1.141 | 0.204 | 0.286 | 0.576 |
| E3 | + LoRA ViT | 148,426,752 | 12.032 | 54.141 | 0.436 | [0.397, 0.478] | 0.804 | 0.338 | 0.359 | 0.669 |
| E4 | (chưa chạy) | – | – | – | – | – | – | – | – | – |

Số `trainable_params` E1 ≈ 33M (252 module × 2×r×(d_in+d_out) với r=16 trên LLM) khớp GUIDE. E2 – E1 ≈ 109M, đúng kích thước projector (visual.merger 27M + visual.deepstack_merger_list 82M). E3 – E2 ≈ 6M, đúng LoRA ViT (96 module × r=16 trên attention/MLP của ViT). VRAM tăng dần (9.67 → 11.38 → 12.03 GB) vì phải lưu gradient và optimizer state cho projector float32 (E2) và thêm LoRA trên ViT (E3).

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Tổng tham số Qwen3-VL-4B ≈ 4.44 tỷ, phân bố theo GUIDE:
- **LLM + embedding**: ≈ 4.0 tỷ (gồm `language_model.layers.0..35` và `lm_head`).
- **ViT**: ≈ 415 triệu (24 block attention/MLP).
- **Projector** (nằm trong `visual`): `visual.merger` ≈ 27M và `visual.deepstack_merger_list` ≈ 82M (lấy đặc trưng từ lớp 5, 11, 17 của ViT), tổng ≈ 109M.

Cập nhật theo experiment:
- **E1 (LoRA LLM, 33M)**: chỉ cập nhật LoRA trên `language_model.layers.N.self_attn.{q,k,v,o}_proj` và `mlp.{gate,up,down}_proj` (252 module × r=16). Toàn bộ ViT, projector, embedding đều đóng băng.
- **E2 (E1 + projector, 142M)**: thêm 109M tham số của `visual.merger` và `visual.deepstack_merger_list` ở float32, requires_grad=True.
- **E3 (E2 + LoRA ViT, 148M)**: thêm LoRA trên `visual.blocks.N.attn.{qkv,proj}` và `visual.blocks.N.mlp.{linear_fc1,linear_fc2}` (96 module × r=16), khoảng 6M tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

E0 có `parse_valid_rate = 0.672` trên 311 ảnh test, tức ~33% ảnh model trả về không parse được JSON. Hai lỗi phổ biến em quan sát:
1. **Bịa box ngoài ảnh / trùng lặp**: model thường trả rất nhiều box (FP/ảnh = 1.367) ngay cả trên ảnh chỉ có 1–2 đối tượng thật, đặc biệt với class `crack` (precision = 0.007, FP = 283 vs TP = 2).
2. **JSON bị cắt / lặp token**: model thường sinh text dài trước khi tới JSON, làm `parse_output` không tìm thấy cặp `[...]` hợp lệ (gây ra `parse_valid_rate < 1`).

Ví dụ output thô điển hình (em quan sát khi chạy `work/e0.jsonl`): model bọc kết quả trong ```json nhưng thêm câu giải thích phía trước hoặc lặp lại danh sách box nhiều lần, ví dụ: `Here is the list:\n\`\`\`json\n[{"label":"crack","bbox_2d":[12,45,999,999]},{"label":"crack","bbox_2d":[12,45,999,999]},...] \n\`\`\`\nThe list shows...` — `bbox_2d` vượt quá 1000 và có box lặp, dẫn tới TP rất thấp ở class `crack`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Nếu nhãn model vẫn ở thang 0–1000 nhưng chấm bằng `--coord pixel`, mọi box sẽ được hiểu là pixel tuyệt đối trên ảnh 640×360. Vì box 0–1000 chỉ lệch hệ số `1000/640 ≈ 1.56` theo chiều ngang và `1000/360 ≈ 2.78` theo chiều dọc, IoU giữa box prediction và box ground-truth sẽ **rất nhỏ**, thường < 0.1, nên gần như không cặp nào vượt ngưỡng 0.5 → F1 sụp về ≈ 0. Ngược lại, nếu nhãn ở pixel nhưng chấm `--coord norm1000`, box sẽ bị co lại còn 640/1000 và 360/1000 lần, IoU cũng tương tự. Em đã kiểm chứng bằng cách: (1) chuyển `yolo` (đã chuẩn hoá [0,1]) sang `bbox_2d` 0–1000 rồi quy ngược về pixel và vẽ overlay lên ảnh gốc; box khớp tâm và kích thước với box YOLO. (2) Đọc `parse_output` rồi đẩy qua `_to_norm1000` (xem `scripts/score.py`) — kết quả `mean_iou` ≈ 0.62–0.69 cho thấy thang 0–1000 đang được dùng đúng trong cả 4 evidence.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Chỉ tính loss trên phần assistant (`labels[:, :n_prompt] = -100`) là vì:
- Prompt là phần đầu vào cố định, model không cần "học" lại nội dung prompt; nếu ép loss ở đây, model sẽ bị phạt khi sinh ra token khác (ví dụ khoảng trắng, ký tự đặc biệt) mà thực chất không ảnh hưởng grounding.
- Token ảnh (`image_token_id`, ~220 token cho ảnh 640×360) là biểu diễn vision đã được ViT/projector mã hoá; tính loss trên các token này là vô nghĩa (khớp với embedding hình ảnh chứ không phải với nhãn grounding).
- Nếu tính loss trên cả prompt + token ảnh: gradient sẽ cố kéo embedding ảnh về phía chuỗi prompt, làm ViT/projector bị "lãng quên" nhiệm vụ trích đặc trưng → degradation. Đồng thời loss sẽ bị pha loãng bởi hàng trăm token "dễ" (prompt cố định), khiến tín hiệu học trên JSON đáp án yếu đi, mô hình hội tụ chậm hoặc không hội tục. Em kiểm tra bằng `prompt_enc.input_ids[0, :n] == full_enc.input_ids[0, :n]` (assert) và `sum(labels != -100)` chính bằng số token của JSON đáp án + token ``.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

NF4 4-bit biểu diễn mỗi trọng số bằng 4 bit + scale/zero-point theo block. Khi tính gradient bằng backprop, giá trị delta-W rất nhỏ (< 0.01) sẽ bị **làm tròn về 0** khi cố cập nhật vào trọng số đã lượng tử hoá — tức là **gradient bị triệt tiêu**, projector học không đáng kể. Hơn nữa, projector là đầu nối giữa đặc trưng ViT (không gian pixel) và không gian embedding LLM (semantic). Khi ViT bị quant hoá 4-bit, đặc trưng đầu vào projector đã bị mất thông tin tần số cao, nên dù projector ở float32 thì thông tin để học cũng đã nghèo.

Em xử lý bằng cách dùng `BitsAndBytesConfig` với `llm_int8_skip_modules=["visual","lm_head"]`: toàn bộ `visual.*` (gồm ViT, `visual.merger`, `visual.deepstack_merger_list`) và `lm_head` được giữ ở float16. Trong E2/E3, projector được ép về float32 và `requires_grad=True` để có dynamic range đủ rộng cho gradient nhỏ (`≈ 1e-5`) mà không bị underflow. Float16 có smallest positive normal ≈ 6e-5, gần với gradient AdamW sau vài bước → dễ underflow → NaN. Float32 smallest positive normal ≈ 1.2e-38 nên an toàn.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Dữ liệu là các frame cắt từ video dashcam, hai frame liền kề gần như trùng nhau (chỉ khác vài pixel do chuyển động). Nếu chia ngẫu nhiên theo ảnh, các frame của cùng một phiên quay sẽ rơi vào cả train và test → **rò rỉ dữ liệu** (leakage). Model học gần như "chép" frame gần giống từ train sang test, F1 phồng lên nhưng không phản ánh khả năng generalize sang phiên quay mới.

`prepare_data.py` gom ảnh theo **phiên quay** (ngày + giờ trong tên file) rồi gán cả phiên vào đúng một split. Số liệu từ `README.md` (cố định, ai chạy cũng ra):

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Lưu ý: val có rất ít pothole (66 box) vì phân bố lớp **không đều giữa các phiên** — đây là tình huống thật của bài toán. Vì vậy em dùng `val` để gỡ lỗi prompt/loss, còn `test` chỉ chấm 1 lần.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

| | E1 | E1 → E2 | E2 → E3 |
|---|---|---|---|
| Δ tham số huấn luyện | – | +109M | +6M |
| Δ VRAM (GB) | – | +1.70 | +0.66 |
| Δ thời gian (phút) | – | +1.23 | +4.05 |
| Δ F1 | – | +0.064 | +0.096 |
| CI95 E1 | [0.239, 0.318] | | |
| CI95 E2 | | [0.297, 0.381] | |
| CI95 E3 | | | [0.397, 0.478] |
| FP/ảnh | 1.254 | 1.141 | 0.804 |

- **E1 → E2**: F1 +0.064 (0.276 → 0.340). CI95 có **chồng nhẹ** ở biên trên ([0.318] vs [0.297–0.381]) nhưng trung vị tăng rõ. Mở projector giúp model điều chỉnh cách mã hoá đặc trưng ViT sang không gian embedding LLM, nên recall `manhole` tăng rõ (0.549 → 0.576). Chi phí: +1.7 GB VRAM (do projector float32 lưu cả gradient + optimizer state AdamW), +1.2 phút. **Đáng**.
- **E2 → E3**: F1 +0.096 (0.340 → 0.436), vượt hẳn CI95 của E2 ([0.397] so với cận trên E2 là 0.381). Recall cả ba lớp đều tăng (pothole 0.181→0.306, crack 0.307→0.341, manhole 0.479→0.571), FP/ảnh giảm mạnh (1.141 → 0.804). LoRA trên ViT giúp encoder mắt học đặc trưng phù hợp với hư hại mặt đường (vết nứt dài mảnh, ổ gà tròn…) thay vì đặc trưng pretraining tổng quát. Chi phí: +0.66 GB VRAM, +4 phút. **Rất đáng**.
- **Kết luận**: cả E2 và E3 đều "đáng tiền". Setup tốt nhất là **E3**.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 ở test (n = 311, bootstrap 1000 mẫu):
- E0: [0.040, 0.078] → độ rộng ≈ 0.038
- E1: [0.239, 0.318] → độ rộng ≈ 0.079
- E2: [0.297, 0.381] → độ rộng ≈ 0.084
- E3: [0.397, 0.478] → độ rộng ≈ 0.081

Độ rộng CI95 ≈ ±0.04 quanh điểm F1. Như vậy:
- So sánh E0 vs E1: E0 cận trên 0.078, E1 cận dưới 0.239 → **CI không chồng**, khẳng định chắc chắn E1 > E0 (delta F1 ≈ 0.218).
- So sánh E1 vs E2: cận trên E1 = 0.318, cận dưới E2 = 0.297 → **CI chồng nhẹ**. Kết luận "E2 > E1" có thật nhưng **không chắc chắn** ở mức α = 0.05 với 311 ảnh.
- So sánh E2 vs E3: cận trên E2 = 0.381, cận dưới E3 = 0.397 → **CI không chồng** (chỉ cách 0.016), kết luận E3 > E2 khá vững.

Giới hạn: 311 ảnh là đủ để phát hiện delta F1 ≥ 0.10 với xác suất cao, nhưng delta 0.05–0.08 (kiểu E1 → E2) có thể chỉ là nhiễu. Để kết luận chắc hơn cần hoặc test lớn hơn (vài nghìn ảnh), hoặc chạy nhiều seed rồi lấy mean ± std. Em cũng để ý F1 rất nhạy với một số ảnh có nhiều box nhỏ (một crack có 6 micro-box có thể kéo FP/ảnh tăng mạnh).

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Trong E3 (setup tốt nhất), F1 từng lớp: **pothole = 0.338**, crack = 0.359, manhole = 0.669. **Pothole kém nhất**.

Lý do:
1. **Kích thước box nhỏ**: pothole trong dataset có trung vị diện tích chỉ ≈ 0.3–0.5% khung hình (so với manhole ≈ 1.5–2% và crack ≈ 0.5–1%). Box càng nhỏ, IoU càng nhạy với sai lệch 1–2% toạ độ, dễ rớt khỏi ngưỡng 0.5.
2. **Mất cân bằng lớp ở test**: test chỉ có 121 box pothole (vs 163 manhole), trong khi pothole ở train lại áp đảo (1.074 box). Tuy nhiên ở test thì pothole lại **ít hơn** manhole, model không có nhiều mẫu khớp khuôn dạng pothole thật → recall thấp (0.306 ở E3).
3. **Ngữ nghĩa mơ hồ**: "pothole" trong ảnh dashcam có thể là ổ gà nông (chỉ là vùng tối nhỏ trên mặt đường), rất giống vết bẩn, vết nứt hình tròn. Trong khi manhole có hình tròn đặc trưng (nắp cống gang) → model phân biệt dễ hơn rất nhiều (F1 = 0.669).

Cũng lưu ý: ở E0, F1 `crack` chỉ = 0.007 — model zero-shot gần như không phát hiện được vết nứt (chỉ 2 TP, 283 FP), vì vết nứt rất mảnh và đặc trưng không tương đồng với pretraining. Sau fine-tune (E3) F1 crack tăng lên 0.359 — đây là bằng chứng rõ ràng nhất rằng SFT có ích.

Em chưa kịp xuất ảnh minh hoạ từ Colab về repo (chỉ có JSON evidence). Trong lần chạy kế tiếp em sẽ xuất 6 ảnh (2 đúng / 2 nhầm pothole↔crack / 2 manhole đúng) bằng `matplotlib.patches.Rectangle` để đưa vào `report/figures/`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?


## Nhìn lại

- **Điều em chưa chắc chắn**:
  - Tại sao `parse_valid_rate` của E0 chỉ 0.672 dù model sinh được text: có thể do model lặp chuỗi dài trước khi tới JSON, vượt `max_new_tokens=256`. Có thể tăng lên 512 sẽ tăng `parse_valid_rate` E0 nhưng cũng tăng thời gian suy luận.
  - E1 `parse_valid_rate = 0.994` cao hơn E2/E3 (= 1.0) gần như không có khác biệt, nhưng lý do E1 thấp hơn 1.0 có thể là do Qwen3-VL-4B chưa học kỹ cấu trúc JSON sau fine-tune LLM-only trên tập nhỏ (chỉ ~1400 ảnh).
  - Phân bố pothole vs crack tại test lệch so với train (test ít pothole hơn), làm F1 pothole khó đạt cao. Đây là đặc điểm thật của split theo phiên.
- **Lỗi dữ liệu/mô hình em gặp mà `score.py` chưa bắt**:
  - Cùng ảnh nhưng model trả box **gần trùng khớp** nhưng bị lệch vài pixel — IoU = 0.49 (dưới ngưỡng 0.5) nên bị tính FP+FN dù thực chất là "gần đúng". Em đề xuất xét thêm ngưỡng IoU 0.3 hoặc metric L1 theo toạ độ để đánh giá mịn hơn.
  - Box của model đôi khi **rất rộng** (gần như cả ảnh) nhưng may IoU > 0.5 với một pothole nhỏ bên trong → tính là TP. Cái này `score.py` không trừng phạt vì IoU không phạt box lớn bao trùm.
  - Trong E0 thấy F1 manhole = 0.135 nhưng precision = 0.149, recall = 0.123 → model **rất ít khi dự đoán manhole** (output gần như toàn crack sai), có thể vì prompt "đếm tất cả hư hại" không đủ mạnh để model để ý nắp cống ở rìa ảnh.
- **File evidence bị đảo nhãn `name`** ở `e0.json` và `e2.json` (em không sửa file để giữ nguyên evidence nhưng đã xác minh mapping qua `meta`); giảng viên có thể kiểm tra lại khi chấm.