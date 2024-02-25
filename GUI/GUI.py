from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui, QtWidgets, QtCore
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model("model.h5")
webcam = cv2.VideoCapture(0)
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
#webcam = cv2.VideoCapture(0)
def extract_features(real_image):
    feature = np.array(real_image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0
def EmotionRecoginization():
    i,im = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im,1.2,5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '% s' % prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                                (0, 0, 255))
        
        return im
    except cv2.error:
        pass
            



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(525, 525)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 380, 161, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.startCamera)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 380, 161, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.stopCamera)

        


        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(90, 70, 350, 300)
        
            
            


        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 30, 291, 21))
        
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        
    def stopCamera(self,MainWindow):
        self.running = False
        
    def startCamera(self, MainWindow):
        self.running = True
        while(self.running):
            im = EmotionRecoginization()
            height,width,channel = im.shape
            bytesPerLine = 3*width
            qImage = QtGui.QImage(im.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(qImage))
            cv2.waitKey(27)
        
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "START"))
        self.pushButton_2.setText(_translate("MainWindow", "STOP"))
        self.label.setText(_translate("MainWindow", "FACIAL EMOTION DETECTION"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
