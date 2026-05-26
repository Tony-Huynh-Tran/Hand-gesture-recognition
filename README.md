# 🤟 Hand Gesture Recognition – Nhận Diện Cử Chỉ Tay

> Đồ án môn **Xử Lý Ảnh** – Khoa Điện – Điện Tử  
> Trường Đại học Công Nghệ Kỹ Thuật TP. Hồ Chí Minh (HUTECH)  
> Giảng viên hướng dẫn: **TS. Dương Minh Thiện**

---

## 📌 Mô tả dự án

Hệ thống nhận diện cử chỉ tay theo thời gian thực dựa trên thị giác máy tính (Computer Vision), kết hợp thư viện **OpenCV**, **MediaPipe** và mô hình học máy huấn luyện bằng **Teachable Machine**.

Hệ thống có khả năng:
- Nhận diện **bảng chữ cái ASL (A–Z)** bằng tay phải
- Nhận diện **chữ số (0–9)** và các ký hiệu đặc biệt (**ENTER**, **SPACE**, **BACKSPACE**) bằng tay trái
- Ghép các ký tự thành từ khóa và **tự động tìm kiếm trên Google** khi nhận cử chỉ ENTER

---

## 👨‍💻 Thành viên nhóm

| Họ và tên | MSSV |
|---|---|
| Huỳnh Trần Phúc Hưng | 23151114 |
| Nguyễn Trung Nguyên | 23151150 |
| Bùi Nguyễn Hải Đăng | 23151078 |

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công cụ / Thư viện |
|---|---|
| Ngôn ngữ lập trình | Python 3.x |
| Xử lý ảnh & video | OpenCV (`cv2`) |
| Phát hiện bàn tay | MediaPipe (qua `cvzone.HandTrackingModule`) |
| Phân loại cử chỉ | Teachable Machine + `cvzone.ClassificationModule` |
| Xử lý ma trận | NumPy |
| Mở trình duyệt | `webbrowser` |

---

## 📁 Cấu trúc thư mục

```
hand-gesture-recognition/
│
├── Model_Phai/
│   └── keras_model.h5          # Mô hình nhận diện tay phải (chữ cái A–Z)
│
├── Model_Trai/
│   └── keras_model.h5          # Mô hình nhận diện tay trái (số 0–9, ENTER, SPACE, BACKSPACE)
│
├── Anh_cho_AI/                 # Thư mục chứa ảnh skeleton dùng để huấn luyện
│   ├── A/
│   ├── B/
│   └── ...
│
├── collect_data.py             # Script thu thập và lưu ảnh skeleton từ webcam
├── recognition.py              # Script nhận diện cử chỉ theo thời gian thực
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install opencv-python
pip install cvzone
pip install mediapipe
pip install numpy
```

> **Lưu ý:** Yêu cầu Python 3.7 trở lên. Khuyến nghị dùng môi trường ảo (venv hoặc conda).

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Thu thập dữ liệu huấn luyện

Chạy script thu thập dữ liệu:

```bash
python collect_data.py
```

- Chỉnh đường dẫn thư mục lưu ảnh trong biến `folder` (ví dụ: `"Anh_cho_AI/A"` để thu thập ảnh cho chữ A)
- Đưa tay vào khung hình, nhấn **`S`** để bắt đầu chụp tự động 1000 ảnh skeleton
- Nhấn **`Q`** để thoát

### Bước 2: Huấn luyện mô hình

1. Truy cập [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Chọn **Image Project** → **Standard image model**
3. Tạo các lớp tương ứng với từng cử chỉ, tải ảnh skeleton đã thu thập lên
4. Nhấn **Train Model**, sau đó **Export Model** → chọn định dạng **Keras**
5. Đặt file `keras_model.h5` vào thư mục `Model_Phai/` (tay phải) hoặc `Model_Trai/` (tay trái)

### Bước 3: Chạy chương trình nhận diện

```bash
python recognition.py
```

**Điều khiển:**
- Dùng **tay phải** → nhận diện chữ cái A–Z
- Dùng **tay trái** → nhận diện số 0–9 và ký hiệu đặc biệt
- Cử chỉ **ENTER** (tay trái) → tìm kiếm từ khóa trên Google
- Cử chỉ **SPACE** → thêm khoảng trắng
- Cử chỉ **BACKSPACE** → xóa ký tự cuối
- Nhấn **`Q`** để thoát chương trình

---

## 🧠 Nguyên lý hoạt động

```
Webcam → Phát hiện bàn tay (MediaPipe)
       → Trích xuất 21 điểm Keypoint
       → Vẽ ảnh skeleton 300×300
       → Đưa vào mô hình Teachable Machine
       → Phân loại cử chỉ → Hiển thị kết quả
```

MediaPipe trích xuất **21 điểm keypoint** trên bàn tay (khớp ngón, đầu ngón, cổ tay), sau đó hệ thống vẽ lại bộ khung xương lên nền trắng để loại bỏ ảnh hưởng của nền và ánh sáng. Ảnh skeleton này được đưa vào mô hình **MobileNet** (Transfer Learning) đã huấn luyện trên Teachable Machine để phân loại.

---

## 📊 Kết quả thực nghiệm

- Hệ thống nhận diện **ổn định** trong điều kiện ánh sáng và nền thông thường
- Nhận diện thành công toàn bộ ký tự A–Z, số 0–9 và các ký hiệu đặc biệt
- Hạn chế: một số ký tự ASL có hình dạng tương đồng (ví dụ: M/N, U/V) dễ bị nhầm lẫn
- Hiệu quả giảm khi ánh sáng quá mạnh hoặc nền quá phức tạp

---

## 🔮 Hướng phát triển

- Mở rộng và đa dạng hóa tập dữ liệu huấn luyện (nhiều người, nhiều góc độ, điều kiện ánh sáng khác nhau)
- Áp dụng mô hình Deep Learning chuyên biệt (TensorFlow / PyTorch) để tăng độ chính xác
- Hỗ trợ nhận diện **cử chỉ động** (gesture động) thay vì chỉ tư thế tĩnh
- Hỗ trợ nhận diện **hai tay đồng thời**
- Tích hợp vào ứng dụng thực tế: hỗ trợ người khiếm thính, điều khiển thiết bị thông minh

---

## 📚 Tài liệu tham khảo

- [Teachable Machine – Google](https://teachablemachine.withgoogle.com/)
- [MediaPipe Hands – Google Research](https://research.google/pubs/mediapipe-a-framework-for-perceiving-and-processing-reality/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Murtaza's Workshop – [Easy hand sign detection | ASL | Computer Vision](https://youtu.be/wa2ARoUUdU8?si=iio8hSSQnEshd0Ix)
- E. A. U. Malahina et al., *Teachable Machine: Real-Time Attendance of Students Based on Open Source System*, ResearchGate, 2023

---

## 📄 Giấy phép

Dự án được thực hiện cho mục đích học tập và nghiên cứu tại HUTECH.  
Tháng 01 năm 2026.
