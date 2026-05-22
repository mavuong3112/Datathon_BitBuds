# Phần 2 — Mô hình Gợi ý (Recommender System)

## 3.3.1 Định nghĩa bài toán

Cho lịch sử quan sát đầy đủ của một người dùng trên Chợ Tốt BĐS đến hết **09/04/2026**, hãy dự đoán **top-10 `item_id` phân biệt** mà người dùng đó sẽ có **tương tác tích cực** trong khoảng **10/04/2026 – 07/05/2026**, xếp hạng từ khả năng cao nhất xuống thấp nhất.

### Định nghĩa tương tác tích cực

Tương tác tích cực được định nghĩa là:

```
fact_user_events.event_type ∈ {view_phone, contact_chat, other_interaction, contact_zalo, contact_sms}
```

### Tập tin được phép dự đoán

Mỗi người dùng test phải dự đoán tối đa **10 `item_id` phân biệt**, lấy từ tập tin có mặt trong `train/dim_listing/`. Các `item_id` nằm ngoài tập này sẽ bị loại bỏ trước khi chấm điểm.

---

## 3.3.2 Nhiệm vụ cốt lõi

- Với mỗi `user_id` trong tập test, dự đoán **top-10 `item_id`** mà người dùng có khả năng cao nhất sẽ liên hệ trong cửa sổ đánh giá.
- **Định dạng output**: `ID, user_id, rank, item_id` (xem chi tiết tại file `sample_submission.csv` trên Kaggle).
- Đề bài **không quy định** kiến trúc, mô hình hay chiến lược cụ thể. Các đội tự lựa chọn cách tiếp cận và trình bày đầy đủ trong báo cáo:
  - Vì sao chọn cách đó
  - Các giả định
  - Các bước tiền xử lí
  - Đặc trưng được sử dụng
  - Cách huấn luyện và đánh giá nội bộ

---

## 3.3.3 Yêu cầu kép: chính xác + lành mạnh

Một hệ khuyến nghị tốt cho marketplace **không chỉ đo bằng accuracy**. BGK kỳ vọng các đội thi có ý thức rõ ràng về tác động của mô hình lên hệ sinh thái Chợ Tốt.

Khuyến khích các đội suy nghĩ và trình bày trong báo cáo về **trade-off** giữa:
- **Độ chính xác**: Contact Rate / Recall / NDCG
- **Marketplace health**: fairness, diversity, freshness, coverage

> Việc minh hoạ trade-off bằng các thử nghiệm so sánh (ví dụ: với và không có cơ chế cân bằng) là một cách trình bày được đánh giá cao — nhưng cụ thể làm thế nào là tự do của mỗi đội.

---

## 3.3.4 Tư duy triển khai

Báo cáo cũng nên thể hiện một mức độ **"tư duy production"** nhất định:
- Nếu mô hình này được triển khai thực tế tại Chợ Tốt thì cần lưu ý gì?
- Mức độ chi tiết và sự tỉnh táo về các vấn đề này là một tín hiệu tích cực với BGK.

---

## 3.3.5 Định dạng file dự đoán

Một file CSV duy nhất, mã hoá **UTF-8 (không BOM)**, đặt tên `submission.csv`. Định dạng **long**:

```
ID,user_id,rank,item_id
```

| Cột       | Kiểu   | Ràng buộc                                                        |
|-----------|--------|------------------------------------------------------------------|
| `ID`      | int    | Unique values và đánh dấu số dòng                                |
| `user_id` | string | Phải có trong `test/test_users.parquet`                           |
| `rank`    | int    | `1 ≤ rank ≤ 10`; mỗi `(user, rank)` là 1 dòng                   |
| `item_id` | string | Phải có trong `train/dim_listing/` (nếu không sẽ bị drop)        |

---

## 3.3.6 Ràng buộc bài nộp

1. **Không leak tương lai.** Chỉ được huấn luyện trên dữ liệu có `date ≤ 09/04/2026` (và `event_ts < 10/04/2026`). Ground truth (10/04 – 07/05) không được public và không được phép sử dụng để train.
2. **Không dùng dữ liệu ngoài.** Mô hình chỉ được huấn luyện từ 4 bảng trong `train/`. Không được sử dụng dataset BĐS bên ngoài, profile người dùng từ nguồn khác, embedding pretrained trên dữ liệu Chợ Tốt rò rỉ, hay API ngoài.
3. **Không reverse-engineer định danh.** Mọi nỗ lực giải ẩn danh đều bị nghiêm cấm.
4. **Số lần nộp:** 5 lượt / đội / ngày UTC.
5. **Tính tái lập (Reproducibility).** Đính kèm toàn bộ mã nguồn. Đặt random seed khi cần thiết. Top winners phải tái tạo lại submission cuối cùng với sai số đủ nhỏ.

---

## Chỉ số đánh giá (Leaderboard Metrics)

### Chỉ số chính: Recall@10

```
Recall@10(u) = |R_u ∩ G_u| / |G_u|
```

Điểm trên bảng xếp hạng là **trung bình Recall@10** trên toàn bộ người dùng test.

### Chỉ số phụ (tie-breaker): NDCG@10

```
DCG@10(u)  = Σ(i=1..10)  𝟙[R_u(i) ∈ G_u] / log₂(i+1)
IDCG@10(u) = Σ(i=1..min(10,|G_u|))  1 / log₂(i+1)
NDCG@10(u) = DCG@10(u) / IDCG@10(u)
```

### Public vs Private leaderboard

- **Public leaderboard**: tính trên chỉ 5% của tập ground truth, chỉ mang tính tham khảo.
- **Private leaderboard**: xếp hạng cuối cùng, được BTC công bố sau khi cuộc thi đóng.

---

## Điều kiện loại bài

Bài nộp sẽ bị loại toàn bộ nếu vi phạm bất kỳ ràng buộc nào sau đây:

1. Sử dụng dữ liệu trong cửa sổ ground truth (`event_ts ≥ 10/04/2026`) dưới mọi hình thức
2. Sử dụng dữ liệu ngoài bộ dữ liệu được cung cấp
3. Không đính kèm mã nguồn hoặc kết quả không thể tái lập
4. Sử dụng dữ liệu sai mục đích

---

## Phân chia dữ liệu

| Tập         | Khoảng thời gian               | Mô tả                                     |
|-------------|--------------------------------|--------------------------------------------|
| `train/`    | 09/11/2025 → 09/04/2026       | Toàn bộ 4 bảng dữ liệu                    |
| `test/`     | —                              | `test_users.parquet`: danh sách `user_id` cần dự đoán |
| `ground_truth/` | 10/04/2026 → 07/05/2026  | Giữ kín, thí sinh không được truy cập       |

---

## Deadline nộp bài

| #  | Hạng mục                    | Kênh nộp            | Deadline                   |
|----|-----------------------------|----------------------|----------------------------|
| 1  | File `submission.csv`       | Kaggle               | 23h59 ngày 21/05/2026      |
| 2  | GitHub + Bài thuyết trình   | Form Vòng Chung kết  | 23h59 ngày 22/05/2026      |
