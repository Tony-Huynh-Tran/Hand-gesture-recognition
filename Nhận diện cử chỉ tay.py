import cv2                                                  #Thư vện xử lí ảnh
from cvzone.HandTrackingModule import HandDetector          #Thư viện tìm bàn tay bằng mediapipe
from cvzone.ClassificationModule import Classifier          #Thư viện phân loại cử chỉ tay bằng model đã train bằng Teachable Machine
import numpy as np                                          #Thư viện toán học xử lí ma trận
import time                                                 #Thư viện thời gian
import webbrowser                                           #Thư viện mở trình duyệt

#   1. Khai báo biến
cap = cv2.VideoCapture(0)                   # Biến luư ảnh từ camera mặc định trên laptop
detector = HandDetector(                       # Cấu hình module tìm bàn tay trong ảnh
    staticMode=False,                          # Bật chế độ nhận diện theo từng frame (không cố định)
    maxHands=1,                                # Số lượng tối đa là nhận diện 1 tay
    modelComplexity=1,                         # 0: model nhẹ, chính xác ít hơn
                                               # 1: model nặng hơn, chính xác cao hơn
    detectionCon=0.5,                          # Ngưỡng tin cậy là 50% để xác nhận là bàn tay                (nhỏ: nhận nhạy, dễ nhận sai
                                                        #                                                     lớn: ít nhận nhầm, dễ mất tay
    minTrackCon=0.5                            # Ngưỡng tin cậy khi track, xác nhận có phải bàn tay cũ khong (nhỏ: tracking tốt nhưng rung;
                                                        #                                                     lớn: tracking mượt nhưng dễ mất tay)
)
current_time = time.time()                  # Cấu hình bộ millis (giống với arduino)
last_time = 0
imgSize = 300                               # Kích thước ảnh cắt ra là 300x300 để tiết kệm bộ nhớ
offset = 20                                 # Khoảng bù kích thước để giữ bàn tay nằm trong khung

labels_Phai = ["A", "B", "C", "D", "E", "F", "G",           # label cho tay phải
               "H", "I", "J", "K", "L", "M", "N",
               "O", "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y", "Z"]

labels_Trai = ["0", "1", "2", "3", "4", "5", "6", "7",      # label cho tay trái
               "8", "9", "ENTER", "SPACE", "BACKSPACE"]

tu_khoa_tim_kiem = ""                                       # Từ khóa để tìm kiếm trên trình duyệt


#   2. Load model
try:        #Load model tay trái
    classifier_Trai = Classifier("Model_Trai/keras_model.h5")
    print("Load xong model tay trái")
except Exception as e:
    print("Lỗi khi load model tay trái")
    exit()

try:        #Load model tay phải
    classifier_Phai = Classifier("Model_Phai/keras_model.h5")
    print("Load xong model tay phải")
except Exception as e:
    print("Lỗi khi load model tay phải")
    exit()



#   3. Hàm vẽ keypoint và xương bàn tay
def draw_keypoint_image(lmList, imgSize=300):
    imgKeypoint = np.ones((imgSize, imgSize, 3), np.uint8) * 255            # Vẽ tạo am trận kích thước 300x300, 3 kênh, 8 bit, mặc định màu trắng
    x_coords = [lm[0] for lm in lmList]                                           # Tạo 1 mảng gồm 21 tọa độ x của keypoint
    y_coords = [lm[1] for lm in lmList]                                           # Tạo 1 mảng gồm 21 tọa độ y của keypoint

    x_min =  min(x_coords)
    x_max =  max(x_coords)
    w =  x_max - x_min

    y_min =  min(y_coords)
    y_max =  max(y_coords)
    h = y_max - y_min


    scale = min((imgSize - 40) / w, (imgSize - 40) / h)     # Tính toán độ scale cho phù hợp với ảnh 300x300
    normalized_lm = []
    for lm in lmList:                                       # Lấy tọa độ 21 Keypoint và ánh x sang tọa độ ảnh crop
        x_norm = int((lm[0] - x_min) * scale + 20)
        y_norm = int((lm[1] - y_min) * scale + 20)
        normalized_lm.append((x_norm, y_norm))
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),                  # Ngón cái
                   (0, 5), (5, 6), (6, 7), (7, 8),                  # Ngón trỏ
                   (0, 9), (9, 10), (10, 11), (11, 12),             # Ngón giữa
                   (0, 13), (13, 14), (14, 15), (15, 16),           # Ngón áp út
                   (0, 17), (17, 18), (18, 19), (19, 20),           # Ngón út
                   (5, 9), (9, 13), (13, 17)]                       # Các đường nối giữa các ngón
    for start, end in connections:
        cv2.line(imgKeypoint, normalized_lm[start], normalized_lm[end], (0, 255, 0), 2)
    for pt in normalized_lm:
        cv2.circle(imgKeypoint, pt, 5, (0, 0, 255), -1)

    return imgKeypoint



#   4. Chương trình chính
while True:
    #   4.1 Tiền xử lí
    success, img = cap.read()                           # Đọc trạng thái mở camera và ảnh camera
    if not success: break                               # Nếu không mở được camera thì baáo lỗi

    img = cv2.flip(img, 1)                      # Lật ngược camera lại như soi gương cho dễ xem
    hands, img = detector.findHands(img, draw=False)    # Tìm hình ảnh bàn tay trong ảnh img, tắt vẽ khung mặc định xung quanh bàn tay
    cao = img.shape[0]                                  # lấy kích thước ảnh
    rong = img.shape[1]

    if hands:                                           # Nếu phát hiện có bàn tay trong ảnh
        hand = hands[0]                                 # Thì lấy bàn tay xuất hiện đầu tiên và có độ tin cậy cao nhất
        lmList = hand['lmList']                         # Lấy danh sách tọa độ 21 keypoint trả ra từ mediapipe
        x, y, w, h = hand['bbox']                       # Lấy tọa độ bounding box bao quanh bàn tay
        hand_type = hand['type']                 # Kiểm tra là tay phải hay tay trái

        if hand_type== "Left":                  # Nếu máy nhận là bàn tay Trái thì là bàn tay Phải (nhận diện chữ cái)
            classifier = classifier_Phai                    # Chọn mô hình phân loại tay Phải
            current_labels = labels_Phai                    # Gán label tay Phải
            color = (0, 255, 0)  # Xanh lá
            cv2.putText(img, "Right hand (text)", (x - offset, y + h + offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:                                           # Nếu máy nhận là bàn tay Phải thì là bàn tay Trái (nhận diện số và các ký hiệu đặc biệt)
            classifier = classifier_Trai                    # Chọn mô hình phân loại tay Trái
            current_labels = labels_Trai                    # Gán label tay Trái
            color = (255, 0, 0)  # Xanh dương
            cv2.putText(img, "Left hand (number)", (x - offset, y + h + offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        imgKeypoint = draw_keypoint_image(lmList, imgSize)  # Vẽ xương ba tay theo Keypoint

        current_time = time.time()                          # Đoán kí tự theo chu kì 1s
        if current_time - last_time > 1:
            prediction, index = classifier.getPrediction(imgKeypoint, draw=False)           # Dự đoán kí tự


            if index < len(current_labels):                                                 # Chỉ xử lí nếu cử chỉ tay nằm trong danh sách đã học
                text = current_labels[index]
                if current_labels == labels_Trai:
                    if text == "ENTER":                                                     # Lệnh ENTER
                        if tu_khoa_tim_kiem != "":
                            url = "https://www.google.com/search?q=" + tu_khoa_tim_kiem
                            webbrowser.open(url)
                            tu_khoa_tim_kiem = ""
                        text = ""
                    elif text == "SPACE":                                                   # Lệnh SPACE
                        text = " "
                    elif text == "BACKSPACE":                                               # Lệnh BACKSPACE
                        tu_khoa_tim_kiem = tu_khoa_tim_kiem[:-1]
                        text = ""

                tu_khoa_tim_kiem = tu_khoa_tim_kiem + text                                  # Viết ừ khóa tìm kiếm
                last_time = current_time                                                    # Reset millis

            else:                                     # Nếu nhận ra kí tự sai thì không xử lí
                pass

        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), color, 2)            # Vẽ bounding box quanh bàn tay
        cv2.rectangle(img, (x - offset, y - offset - 45), (x + w + offset, y - offset), color, cv2.FILLED)  # Vẽ vùng để viết chữ lên

        if index < len(current_labels):
            # Vẽ kí tự nhận diện
            cv2.putText(img, f"{current_labels[index]}", (x - offset + 5, y - offset - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Vẽ hình xương Keypoint
        cv2.imshow("Keypoint", imgKeypoint)

    # Vẽ thanh tìm kiếm
    cv2.putText(img, f"Search: {tu_khoa_tim_kiem}", (10, cao - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Hand Sign Recognition", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Happy New Year!!!")