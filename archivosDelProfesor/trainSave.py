import face_recognition
import cv2
import os
import pickle

print(cv2.__version__)

Encodings = []
Names = []

image_dir = '/home/jetson/Documents/JETBOT/FaceRecognizer/Train'
for root, dirs, files in os.walk(image_dir):
    print(files)
    for file in files:
        path = os.path.join(root, file)
        print(path)
        name = os.path.splitext(file)[0]
        print(name)
        
        # Cargar la imagen y encontrar las codificaciones de la cara
        person = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(person)
        
        # Verificar si se encontraron codificaciones
        if encodings:
            encoding = encodings[0]  # Obtener la primera codificación
            Encodings.append(encoding)
            Names.append(name)
        else:
            print(f"No se encontraron caras en la imagen: {file}")

print(Names)

# Guardar las codificaciones y nombres en un archivo pickle
with open('train.pkl', 'wb') as f:
    pickle.dump(Names, f)
    pickle.dump(Encodings, f)
