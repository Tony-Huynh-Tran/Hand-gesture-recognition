import cv2                                              # Thư viện xử lí ảnh
from cvzone.HandTrackingModule import HandDetector      # Thư viện tìm bàn tay bằng mediapipe
import numpy as np                                      # Thư viện toán học xử lí ma trận
import os                                               # Thư viện thao tác với hệ thống file

#   1. Khai báo biến
camera = cv2.VideoCapture(0)                   # Biến luư ảnh từ camera mặc định trên laptop
detector = HandDetector(                       # Cấu hình module tìm bàn tay trong ảnh
    staticMode=False,                          # Bật chế độ nhận diện theo từng frame (không cố định)
    maxHands=1,                                # Số lượng tối đa là nhận diện 1 tay
    modelComplexity=1,                         # 0: model nhẹ, chính xác ít hơn
                                               # 1: model nặng hơn, chính xác cao hơn
    detectionCon=0.5,                          # Ngưỡng tin cậy là 50% để xác nhận là bàn tay             (nhỏ: nhận nhạy, dễ nhận sai
                                                        #                                                  lớn: ít nhận nhầm, dễ mất tay
    minTrackCon=0.5                            # Ngưỡng tin cậy khi track, xác nhận có phải bàn tay cũ ko (nhỏ: tracking tốt nhưng rung;
                                                        #                                                  lớn: tracking mượt nhưng dễ mất tay)
)
offset = 20                                   # Khoảng bù kích thước để giữ bàn tay nằm trong khung
counter = 0                                   # Đếm số lượng ảnh
folder = r"D:\Project\Mon_xu_li_anh\Bao_cao_do_an\Anh_cho_AI\A"
os.makedirs(folder, exist_ok=True)            # Kiểm tra xem có thư mục này chưa, có thì thôi
                                              #                                  chưa có thì tự tạo




#   2. Hàm vẽ keypoint và xương bàn tay
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


    scale = min((imgSize - 2*offset) / w, (imgSize - 2*offset) / h)     # Tính toán độ scale cho phù hợp với ảnh 300x300
    normalized_lm = []
    for lm in lmList:                                       # Lấy tọa độ 21 Keypoint và ánh x sang tọa độ ảnh crop
        x_norm = int((lm[0] - x_min) * scale + offset)
        y_norm = int((lm[1] - y_min) * scale + offset)
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

#   3. Hàm vẽ khung và chữ lên ảnh
def draw_box_and_text(img, hand_info):
    x, y, w, h = hand_info['bbox']              # Lấy tọa độ bounding box bao quanh bàn tay
    hand_type = hand_info['type']               # Kiểm tra là tay phải hay tay trái

    if hand_type == "Left":
        text_info = "Phai"
    else:
        text_info = "Trai"

    # Vẽ hình chữ nhật (Khung xanh lá)
    p1 = (x - offset, y - offset)
    p2 = (x + w + offset, y + h + offset)
    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    # Viết chữ lên trên khung
    cv2.putText(img, text_info, (x - offset, y - offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

#   4. Vòng lập chính
while camera.isOpened():
    success, img = camera.read()
    if not success: break

    img = cv2.flip(img, 1)                    # Lật lại cho dễ nhìn
    hands, img = detector.findHands(img, draw=False)  # Tắt draw mặc định để mình tự vẽ

    if hands:                                         # Nếu phát hiện có bàn tay trong ảnh
        hand = hands[0]                               # Thì lấy bàn tay xuất hiện đầu tiên và có độ tin cậy cao nhất
        lmList = hand['lmList']                       # Lấy danh sách tọa độ 21 keypoint trả ra từ mediapipe
        img = draw_box_and_text(img, hand)            # Vẽ khung và nhận diện bàn tay

        # Vẽ xương lên màn hình phụ (imgKeypoint)
        imgKeypoint = draw_keypoint_image(lmList, 300)
        cv2.imshow("Hand keypoint", imgKeypoint)

    cv2.imshow("Main Camera", img)
    key = cv2.waitKey(1) & 0xFF

    # --- BẤM S ĐỂ LƯU DỮ LIỆU ---
    if key == ord("s") or key == ord("S"):
        if hands:
            for i in range(1000):  # Chụp liên tục 1000 tấm
                success, img_capture = camera.read()
                if not success: break

                img_capture = cv2.flip(img_capture, 1)
                hands_cap, img_capture = detector.findHands(img_capture, draw=False)

                if hands_cap:
                    hand_cap = hands_cap[0]
                    lmList = hand_cap['lmList']

                    # Vẽ 2 ảnh lên
                    img_capture = draw_box_and_text(img_capture, hand_cap)
                    imgSave = draw_keypoint_image(lmList, 300)

                    # Lưu file ảnh xương
                    cv2.imwrite(f'{folder}/Keypoint_{counter}.jpg', imgSave)
                    counter += 1

                    # Hiển thị tiến độ
                    cv2.putText(img_capture, f'Counting: {i + 1}/1000 (Total: {counter})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Main Camera", img_capture)
                    cv2.imshow("Hand keypoint", imgSave)

                    # Thoát nhanh nếu bấm q trong lúc đang chụp
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            print("Đã chụp xong 1000 tấm!")

    if key == ord('q'): break

camera.release()
cv2.destroyAllWindows()