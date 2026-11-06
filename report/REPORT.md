# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
 {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.51087
  },
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về toàn bộ bức ảnh, vì trường `task` có giá trị `image_classification`. Mô hình YOLO11 Classification dự đoán một lớp cho cả ảnh, không xác định vị trí hay số lượng từng vật thể trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list) có sẵn trong taxonomy đi kèm checkpoint của mô hình, không phải do mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả 3 vì `class_id` là mã số, `class_name` là tên lớp, còn `taxonomy_name` cho biết lớp đó thuộc danh sách nào. Nhờ vậy, mình biết chính xác AI đang dự đoán gì và tránh nhầm lẫn.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí chọn một lớp cho cả ảnh, ví dụ dựa vào vật thể chính, nổi bật nhất hoặc chiếm diện tích lớn nhất. Nếu có nhiều vật thể nổi bật như nhau thì cần có quy tắc xử lý thống nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là điểm dự đoán của AI, không đảm bảo kết quả đúng. Cần kiểm tra và xác nhận nhãn theo guideline để có Ground Truth.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 45,
    "class_name": "bowl",
    "score": 0.700724,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      155.02,
      168.66,
      182.38,
      184.18
    ],
    "bbox_width": 27.36,
    "bbox_height": 15.52
  },
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái: (155.02, 168.66)
góc dưới-phải: (182.38, 184.18)
rộng 27.36 pixel và cao 15.52 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ở ngưỡng 0.20 có 17 prediction
Ở ngưỡng 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, reviewer cần kiểm tra thêm các hộp dự đoán, xem AI có nhận diện sai vật thể, khoanh sai vị trí hoặc tạo hộp trùng lặp hay không.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc: Khi vẽ Box, phải vẽ khung ôm sát vật thể, không cắt mất phần nhìn thấy và không lấy thừa nền xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vật bị che khuất hoặc cắt mép: Người ở mép trái ảnh `kitchen` (`person 0.61`) bị cắt mất phần lớn cơ thể.

Guideline hay Escalation: Em chọn Escalation (hỏi Lab Coach) nếu guideline chưa có quy tắc xử lý.

Vì sao: Vì chỉ nhìn thấy một phần cơ thể nên khó xác định cách vẽ Bounding Box chính xác, cần hỏi Lab Coach để tránh gán nhãn sai.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
"sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-009",
    "class_id": 0,
    "class_name": "person",
    "score": 0.432257,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      0.3,
      263.27,
      59.5,
      309.92
    ],
    "polygon_point_count": 65,
    "polygon_xy": [
      [
        1.0,
        264.0
      ],
      [
        1.0,
        309.0
      ],
      [
        5.0,
        309.0
      ],
      [
        6.0,
        308.0
      ],
      [
        15.0,
        308.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho biết đường viền và hình dạng vật thể chi tiết hơn, còn Bounding Box chỉ xác định vùng hình chữ nhật bao quanh vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` dùng để phân biệt từng vật thể riêng biệt trong ảnh. Nó khác `class_id` vì không xác định loại vật thể, và khác `tracking_id` vì không dùng để theo dõi vật thể qua nhiều khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc: Khi vẽ Mask, đường viền phải bám sát phần nhìn thấy của vật thể, không lấy thừa vùng nền và không bỏ sót phần vật thể nhìn thấy.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Vị trí khó xác định: Chiếc bát `bowl 0.55` ở góc dưới bên trái, nằm sát các vật thể khác nên khó xác định đường viền.

Chọn: Escalation (hỏi Lab Coach) nếu guideline chưa quy định cách xử lý.

Vì sao: Vì các vật thể nằm sát nhau, dễ vẽ nhầm vùng Mask. Cần hỏi Lab Coach để thống nhất cách gán nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn cho cả ảnh | `traffic`: AI chọn `cab` (0.510915), dù ảnh có nhiều loại xe | Chọn nhãn theo guideline | Kiểm tra nhãn đúng quy tắc |
| Phát hiện vật thể | Một lớp và box cho mỗi vật thể | `kitchen`: `person 0.61` bị cắt mép | Vẽ box theo guideline | Kiểm tra vị trí, kích thước box |
| Instance segmentation | Một lớp và mask cho mỗi vật thể | `kitchen`: `bowl 0.55` sát vật khác | Vẽ mask theo biên vật thể | Kiểm tra mask có lấn vùng khác không ||

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ảnh công khai được phép, không đưa dữ liệu cá nhân hoặc dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng xử lý và báo ngay cho Lab Coach để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
