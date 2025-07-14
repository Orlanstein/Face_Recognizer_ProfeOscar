import face_recognition
import cv2
import os
import pickle

print(cv2.__version__)

Encodings = []
Names = []

# Carga de nombres y codificaciones desde el archivo pickle
with open('/home/orin/Downloads/pyPro-20250701T235910Z-1-001/pyPro/pyPro/FaceRecognizer/train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX
#cam = cv2.VideoCapture(0)  # O 1, dependiendo de tu configuración+
# Configuración de la cámara USB
dispW = 640
dispH = 480
flip = 2
pipeline = f"v4l2src device=/dev/video0 ! video/x-raw, width={dispW}, height={dispH}, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"

# Inicializa la captura de video con el pipeline
cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frameSmall = cv2.resize(frame,(0,0),fx = .33, fy = .33)
    frameRGB = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    face_positions = face_recognition.face_locations(frameRGB, model='hog')
    all_encodings = face_recognition.face_encodings(frameRGB, face_positions)

    for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
        
        top = top*3
        right = right*3
        left = left*3
        bottom = bottom*3
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 6), font, 0.75, (0, 0, 255), 2)

    cv2.imshow('Picture', frame)
    cv2.moveWindow('Picture', 0, 0)

    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cam.release()
cv2.destroyAllWindows()
