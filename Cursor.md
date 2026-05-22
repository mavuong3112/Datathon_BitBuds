Lưu ý sau đây là 1 số định hướng xử lý dữ liệu.

## Time coverage

- Xử lý **bất đồng bộ** giữa `dim_listing.posted_date` và cửa sổ `fact_user_events` / `fact_listing_snapshot` (dim có thể sớm hơn facts).

## `is_login` — login vs non-login (đọc kỹ trước khi lọc)

README: với **non-login**, `user_id` **đổi theo từng session** → không dùng `user_id` để cohort user marketing / repeat-user theo ngày.

**Quy tắc tách bạch:**

| Mục đích phân tích | Đơn vị quan sát | Lọc `is_login`? | Ghi chú |
|-------------------|-----------------|-----------------|--------|
| Browse, pageview, funnel xem tin | **`session_id`** (và/hoặc `device`) | **Không** — giữ **cả login + non-login** | Đo “nhu cầu / exposure” trên toàn marketplace |
| So sánh login vs non-login | `session_id` × `is_login` | **Không** — tách nhóm | Non-login thường volume lớn hơn |
| Explicit contact / lead / doanh thu | `user_id` (login) | **Chỉ `is_login = 'login'`** | Outcome gắn tài khoản |
| **`contact_chat`** | `user_id` | **Bắt buộc login** | README: chat **chỉ** tạo khi đã login — không có non-login chat thật |
| `view_phone`, `contact_zalo`, `contact_sms` | `user_id` | **Chỉ login** (cho metric user-level) | Cùng lớp explicit / lead |
| Repeat multi-day, time-to-contact **theo user** | `user_id` | **Chỉ login** | Non-login không có `user_id` ổn định |
| Repeat trong phiên, search refinement | `session_id` | **Cả hai** | Refinement / path trong session không cần login |

**Tóm lại:** Khi “đụng” tới **quan sát hành vi + `event_type` + tương tác tin** → **phải nhìn cả non-login** (session-level). Khi đo **contact / chat / conversion gắn user** → **chỉ login**; riêng **`contact_chat` luôn implied login**.

**Sai lầm thường gặp:** Lọc `is_login = 'login'` cho toàn bộ pipeline → làm **mất phần lớn browse** và đánh giá sai “đào mỏ” / friction (chỉ thấy người đã đủ quan tâm để đăng nhập).

## `dwell_time_sec`

- Nên **chia 1000** (có thể đang lưu ms).

## Outcome / doanh thu / positive interaction

Doanh thu khi liên hệ thực sự: xem SĐT, chat, Zalo, SMS.

`event_type` **tích cực (datathon):**
`view_phone`, `contact_chat`, `other_interaction`, `contact_zalo`, `contact_sms`

- **Explicit contact (lead):** 4 loại đầu + SMS — **không** gom `other_interaction` (~95% `surface=ad_view`) khi đo contact rate / concentration.

## Category (định nghĩa đúng)

| Mã | Mô tả |
|----|--------|
| 1010 | Căn hộ / Chung cư |
| 1020 | Nhà ở |
| 1030 | Văn phòng / Mặt bằng |
| 1040 | Đất |
| 1050 | Phòng trọ |

- Cột theo category: `13b_dim_listing_eda_role_matrix.csv`

## QA / tiền xử lý (checklist)

**Nhóm A — Integrity:** duplicate `item_id`, contacts>views, `is_contact` vs `event_type`, events.category vs dim, orphan items  

**Nhóm B — Date/Time:** future leak, expired<posted, timezone, `date` vs `event_ts`  

**Nhóm C — Numeric:** dwell extreme, views=0, `area_sqm`, bedrooms/bathrooms/floors/width, position dtype  

**Nhóm D — Categorical:** `price_bucket` edge cases, category–adtype mismatch, whitespace, `purchased`  

**Nhóm E — User behavior:** non-login `user_id` đổi theo session; session có thể trộn `is_login` — **dùng `session_id` cho funnel, `user_id` chỉ khi login**
