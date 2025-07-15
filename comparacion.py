import cv2

# Haar Cascade
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
frame = cv2.imread("personas.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces_haar = haar_cascade.detectMultiScale(gray, 1.1, 4)

# HOG + SVM (usando Dlib)
import dlib
hog_face_detector = dlib.get_frontal_face_detector()
faces_hog = hog_face_detector(gray)

# CNN (usando Dlib)
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
faces_cnn = cnn_face_detector(frame, 1)
