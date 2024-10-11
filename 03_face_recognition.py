import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#khởi tạo bộ đếm id
id = 0

# tên liên quan đến id: ví dụ ==> Marcelo: id=1, v.v.
names = ['None','VanNhan','Phat','Nhut','Nguyen','Vuot']

# Khởi tạo và bắt đầu quay video thời gian thực
cam = cv2.VideoCapture(0)
cam.set(3, 640) #  độ rộng video
cam.set(4, 480) # độ dài video

# Xác định kích thước cửa sổ tối thiểu để được nhận dạng là khuôn mặt
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Lật theo chiều dọc

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
       
        # Kiểm tra xem độ tin cậy có thấp hơn họ 100 ==> "0" có khớp hoàn hảo không
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Nhấn 'ESC' để thoát video
    if k == 27:
        break

# Dọn dẹp một chút
print("\n [THÔNG TIN] Thoát khỏi chương trình và dọn dẹp nội dung")
cam.release()
cv2.destroyAllWindows()