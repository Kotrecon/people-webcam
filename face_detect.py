# подключаем библиотеку компьютерного зрения
import cv2
# библиотека для вызова системных функций
import os

# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(path+r'/trainer/trainer.yml')
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# получаем доступ к камере
cam = cv2.VideoCapture(0)
# настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX

# запускаем цикл
while True:
    # получаем видеопоток
    ret, im =cam.read()
    # переводим его в ч/б
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # определяем лица на видео
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    # перебираем все найденные лица
    for(x,y,w,h) in faces:
        # получаем id пользователя
        nbr_predicted,coord = recognizer.predict(gray[y:y+h,x:x+w])
        # рисуем прямоугольник вокруг л≈ица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # если мы знаем id пользователя
        if(nbr_predicted==1):
             # подставляем вместо него имя человека
             nbr_predicted='Ermakov'
        if (nbr_predicted == 2):
            # подставляем вместо него имя человека
            nbr_predicted = 'Damir'
        # добавляем текст к рамке
        cv2.putText(im,str(nbr_predicted), (x,y+h),font, 1.1, (0,255,0))
        # выводим окно с изображением с камеры
        cv2.imshow('Face recognition',im)
        # делаем паузу
        cv2.waitKey(10)