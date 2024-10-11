import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # độ độ rộng video
cam.set(4, 480) # độ chiều cao video

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Đối với mỗi người, hãy nhập một id khuôn mặt bằng số
face_id = input('\n nhập id người dùng cuối nhấn <enter> ==>  ')

print("\n [THÔNG TIN] Đang khởi tạo tính năng chụp khuôn mặt. Nhìn vào camera và chờ đợi ...")
# Khởi tạo số lượng khuôn mặt lấy mẫu riêng lẻ
count = 0
1
while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # lật hình ảnh video theo chiều dọc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Lưu ảnh đã chụp vào thư mục bộ dữ liệu
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Nhấn 'ESC' để thoát video
    if k == 27:
        break
    elif count >= 100: # Lấy mẫu 30 khuôn mặt và dừng video
         break

# Dọn dẹp một chút
print("\n [THÔNG TIN] Thoát khỏi chương trình và dọn dẹp nội dung")
cam.release()
cv2.destroyAllWindows()
