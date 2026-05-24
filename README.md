<img width="2382" height="1346" alt="image" src="https://github.com/user-attachments/assets/9404f47d-e7c9-440b-9975-1b719cbfc90e" />
<img width="2370" height="1334" alt="image" src="https://github.com/user-attachments/assets/fe4a54a1-4f7d-41bd-ad59-d2de28afe5b4" />
<img width="2366" height="1336" alt="image" src="https://github.com/user-attachments/assets/ced09c36-2683-4b0e-bd1d-07f31378bcab" />
<img width="2364" height="1326" alt="image" src="https://github.com/user-attachments/assets/980f7f02-66e6-4b48-8c26-20ffd45936e7" />
<img width="2368" height="1324" alt="image" src="https://github.com/user-attachments/assets/9a6d3651-5264-4789-a131-14dbffdaa03e" />
<img width="2364" height="1336" alt="image" src="https://github.com/user-attachments/assets/a0631874-0b35-46e6-9896-024bc36ce053" />
<img width="2364" height="1334" alt="image" src="https://github.com/user-attachments/assets/df0eb2a0-e26c-4fa4-b04f-9e4ceefeeaf2" />
<img width="2360" height="1308" alt="image" src="https://github.com/user-attachments/assets/95170dcd-2f89-4e25-bc4d-94c33c3e4892" />
<img width="2368" height="1296" alt="image" src="https://github.com/user-attachments/assets/9b708d9f-c556-46ae-9612-4e5903df1826" />
<img width="2348" height="1316" alt="image" src="https://github.com/user-attachments/assets/8237583f-1626-4116-ba69-5ef5078039b9" />
<img width="2342" height="1322" alt="image" src="https://github.com/user-attachments/assets/9f8114c8-ba53-41a9-945f-4b470a958811" />

https://canva.link/8x8dak5xfnpdjak

## Giới thiệu

Chợ Tốt là một trong những sàn thương mại điện tử C2C hàng đầu Việt Nam, với mảng bất động sản (BĐS) bao gồm các phân khúc cho thuê, căn hộ chung cư, nhà ở, đất nền và dự án phát triển mới. Mô hình kinh doanh của nền tảng dựa trên **tạo lead (lead-generation)**: doanh thu phát sinh khi người dùng tiến hành liên hệ thực sự với người bán — xem số điện thoại, mở cuộc trò chuyện, gọi qua Zalo hay nhắn tin SMS.

Bộ dữ liệu là bản chụp star schema thực tế của hệ thống Chợ Tốt trong giai đoạn 09/11/2025 – 07/05/2026, đã được ẩn danh hoàn toàn (mọi định danh đều là chuỗi SHA-256 hex). Tổng dung lượng nguyên gốc ~52 GB / 949 file Parquet. Bản phát hành cho thí sinh đã được cắt theo trục thời gian.

---

## Định nghĩa tương tác tích cực (positive interaction)

Một sự kiện trong `fact_user_events` được xem là tích cực khi và chỉ khi:

`event_type ∈ {view_phone, contact_chat, other_interaction, contact_zalo, contact_sms}`

> **Lưu ý quan trọng:** Đây chính là mục tiêu dự đoán của cuộc thi: 5 loại sự kiện này thể hiện ý định mua/thuê thực sự, có giá trị thương mại cao và gắn liền với doanh thu của nền tảng. Các sự kiện duyệt trang (pageview) chỉ là tín hiệu nền (browsing noise) và không được tính là tích cực.

---

## Phân chia dữ liệu cho bài toán khuyến nghị

- **train/:** 09/11/2025 → 09/04/2026 (toàn bộ 4 bảng)
- **test/test_users.parquet:** danh sách `user_id` cần dự đoán
- **ground_truth/:** 10/04/2026 → 07/05/2026 — giữ kín, thí sinh không được truy cập

---

## Tổng quan các bảng dữ liệu


| #   | Bảng                           | Lớp          |
| --- | ------------------------------ | ------------ |
| 1   | dim_listing                    | Dimension    |
| 2   | fact_listing_snapshot          | Fact (daily) |
| 3   | fact_post_contact_interactions | Fact (daily) |
| 4   | fact_user_events               | Clickstream  |
| 5   | test_users.parquet             | Test set     |


---

## Bảng Dimension

### ▶ dim_listing — Danh mục tin đăng BĐS

Khoá chính: `item_id`. Một dòng tương ứng với một tin đăng.


| Cột                       | Kiểu  | Mô tả                                                                         |
| ------------------------- | ----- | ----------------------------------------------------------------------------- |
| **item_id**               | str   | Khoá chính (SHA-256)                                                          |
| **seller_id**             | str   | Mã người bán (SHA-256)                                                        |
| **category**              | int   | Mã danh mục: 1010/1020/1030/1040/1050                                         |
| **title**                 | str   | Tiêu đề tin (tiếng Việt, có thể chứa emoji)                                   |
| **seller_type**           | str   | agent (môi giới) / private (cá nhân)                                          |
| **ad_type**               | str   | sell (bán) / let (cho thuê)                                                   |
| **ad_status**             | str   | Trạng thái tin                                                                |
| **area_sqm**              | float | Diện tích sử dụng (m²)                                                        |
| **bedrooms, bathrooms**   | float | Số phòng ngủ, phòng tắm                                                       |
| **floors, width_m**       | float | Số tầng, mặt tiền (m)                                                         |
| **direction**             | str   | Hướng nhà tiếng Việt: Đông/Tây/Nam/Bắc + tổ hợp                               |
| **legal_status**          | str   | Tình trạng pháp lý (vd. Đã có sổ, Sổ hồng riêng)                              |
| **house_type**            | str   | Loại nhà (vd. Nhà ngõ, hẻm, Nhà mặt phố)                                      |
| **furnishing**            | str   | Nội thất (vd. Nội thất đầy đủ, Nhà trống)                                     |
| **city_name**             | str   | Tên tỉnh/thành (tiếng Việt)                                                   |
| **district_name**         | str   | Tên quận/huyện (tiếng Việt)                                                   |
| **ward_name**             | str   | Tên phường/xã (tiếng Việt)                                                    |
| **project_id**            | str   | Chỉ có với tin thuộc dự án                                                    |
| **price_bucket**          | str   | Khoảng giá (chuỗi tiếng Việt, vd. 3 tỷ - 5 tỷ, 5 triệu/tháng - 7 triệu/tháng) |
| **images_count**          | float | Số ảnh trong tin                                                              |
| **posted_date**           | date  | Ngày đăng tin                                                                 |
| **expected_expired_date** | date  | Ngày hết hạn dự kiến                                                          |


### ▶ Mã danh mục category


| Mã       | Mô tả               |
| -------- | ------------------- |
| **1010** | Căn Hộ / Chung cư   |
| **1020** | Nhà ở               |
| **1030** | Văn Phòng, Mặt Bằng |
| **1040** | Đất                 |
| **1050** | Phòng trọ           |


---

## Bảng Fact

### ▶ fact_listing_snapshot — Hiệu năng tin đăng theo ngày

Một dòng = một (`item_id`, `date`).


| Cột                  | Kiểu  | Mô tả                         |
| -------------------- | ----- | ----------------------------- |
| **item_id**          | str   | FK → dim_listing              |
| **date**             | date  | Ngày chụp                     |
| **views_24h**        | float | Lượt xem trong 24h            |
| **contacts_24h**     | float | Lượt liên hệ trong 24h        |
| **listing_age_days** | float | Tuổi tin tại ngày chụp (ngày) |


### ▶ fact_post_contact_interactions — Tổng hợp tương tác user × tin theo ngày

Một dòng = một (`user_id`, `item_id`, `date`).

*Lưu ý:* Tên thư mục là `fact_post_contact_interactions` nhưng tên file gốc lại bắt đầu bằng `datathon_fact_user_ad_interactions-`*. Đây là sự không khớp cố ý giữ nguyên từ pipeline gốc; 


| Cột                    | Kiểu   | Mô tả                                                                                                                       |
| ---------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| **user_id**            | string | Mã người dùng                                                                                                               |
| **item_id**            | string | FK → dim_listing                                                                                                            |
| **date**               | date   | Ngày                                                                                                                        |
| **adview_count**       | float  | Số lượt xem tin                                                                                                             |
| **lead_count**         | float  | Số lượt lộ số điện thoại / email                                                                                            |
| **chat_message_count** | float  | Số tin nhắn chat                                                                                                            |
| **chat_turn_count**    | float  | Số lượt qua lại trong chat                                                                                                  |
| **chat_lead**          | float  | Cờ lead phát sinh từ chat                                                                                                   |
| **purchased**          | bool   | Dự đoán của Chợ Tốt về việc tin đăng đã có người mua/thuê được hay chưa. Đây là nhãn dự đoán nội bộ — có thể đúng hoặc sai. |
| **category**           | int    | Mã danh mục tin                                                                                                             |


### ▶ fact_user_events — Clickstream thô

Bảng lớn nhất, chiếm phần lớn dung lượng toàn bộ tập dữ liệu. Mỗi dòng = một hành vi của người dùng.


| Cột                | Kiểu      | Mô tả                                                                                                     |
| ------------------ | --------- | --------------------------------------------------------------------------------------------------------- |
| **is_login**       | string    | login / non-login                                                                                         |
| **user_id**        | string    | Mã người dùng (lưu ý: với non-login, user_id thay đổi theo từng phiên), người dùng phải login mới có chat |
| **session_id**     | string    | Mã phiên                                                                                                  |
| **event_id**       | string    | Mã sự kiện                                                                                                |
| **item_id**        | string    | FK → dim_listing                                                                                          |
| **city_name**      | string    | Tên TP của tin (tiếng Việt)                                                                               |
| **category**       | int       | Mã danh mục                                                                                               |
| **event_type**     | string    | Loại sự kiện (xem bảng dưới)                                                                              |
| **query**          | string    | Truy vấn tìm kiếm (chỉ có với pageview từ search)                                                         |
| **event_ts**       | timestamp | Thời điểm sự kiện                                                                                         |
| **surface**        | string    | Bề mặt UI nơi sự kiện diễn ra                                                                             |
| **position**       | float     | Vị trí trong feed / kết quả tìm                                                                           |
| **device**         | string    | Desktop / MSite / iOS / Android                                                                           |
| **dwell_time_sec** | float     | Thời gian dừng trên trang tin (giây)                                                                      |
| **is_contact**     | int       | Cờ 1 nếu sự kiện là liên hệ                                                                               |
| **date**           | date      | Ngày của event_ts                                                                                         |


### Các giá trị event_type


| Giá trị               | Ý nghĩa                                |
| --------------------- | -------------------------------------- |
| **pageview**          | Xem trang tin                          |
| **other_interaction** | **TÍCH CỰC**                           |
| **view_phone**        | **TÍCH CỰC** — bấm xem số điện thoại   |
| **contact_chat**      | **TÍCH CỰC** — mở phiên chat thuộc app |
| **contact_zalo**      | **TÍCH CỰC** — nhắn Zalo               |
| **contact_sms**       | **TÍCH CỰC** — nhắn SMS                |


---

## Quan hệ giữa các bảng


| Quan hệ                                          | Cardinality                 |
| ------------------------------------------------ | --------------------------- |
| **dim_listing ↔ fact_listing_snapshot**          | 1 : nhiều (theo ngày)       |
| **dim_listing ↔ fact_post_contact_interactions** | 1 : nhiều (theo user, ngày) |
| **dim_listing ↔ fact_user_events**               | 1 : nhiều (theo sự kiện)    |
| **user_id (events) ↔ user_id (interactions)**    | nhiều : nhiều               |


---

## Lưu ý chung

> Dữ liệu được giữ nguyên trạng (raw). Thí sinh cần tự đánh giá chất lượng dữ liệu (missing values, outlier, encoding. . . ) và đề xuất phương pháp tiền xử lí phù hợp với bài toán của mình.

