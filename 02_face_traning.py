

import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến cơ sở dữ liệu hình ảnh khuôn mặt
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# chức năng lấy hình ảnh và dữ liệu nhãn
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # chuyển đổi nó sang thang độ xám
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Đào tạo khuôn mặt. Nó sẽ mất một vài giây. Chờ đợi ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu mô hình vào trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # nhận dạng.save() hoạt động trên Mac nhưng không hoạt động trên Pi
# In số mặt đã được huấn luyện và kết thúc chương trình
print("\n [INFO] {0} khuôn mặt đã được đào tạo. Thoát khỏi chương trình".format(len(np.unique(ids))))