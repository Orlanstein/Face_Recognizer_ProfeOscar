import cv2
import time
import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import face_recognition  # Added missing import

# Configuración inicial
DURACION_PRUEBA = 10  # segundos
UMBRAL_CONFIANZA = 0.6  # para reconocimiento facial
RESULTADOS_DIR = "resultados_video"
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Cargar encodings conocidos
def cargar_encodings(archivo_pkl):
    with open(archivo_pkl, 'rb') as f:
        nombres = pickle.load(f)
        encodings = pickle.load(f)
    return nombres, encodings

try:
    nombres_conocidos, encodings_conocidos = cargar_encodings('train.pkl')
except FileNotFoundError:
    print("Error: No se encontró el archivo train.pkl")
    exit()

# Inicializar detectores
detector_mtcnn = MTCNN()
detector_hog = cv2.HOGDescriptor()
detector_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
detector_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configurar cámara (o video)
dispW, dispH = 640, 480
try:
    # Try different capture methods
    cam = cv2.VideoCapture(0)  # First try default camera
    if not cam.isOpened():
        # If default fails, try GStreamer pipeline
        pipeline = f"v4l2src device=/dev/video0 ! video/x-raw, width={dispW}, height={dispH}, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cam.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    
    # Set resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
except Exception as e:
    print(f"Error al inicializar la cámara: {e}")
    exit()

# Estructuras para almacenar resultados
resultados = []
tiempo_inicio = time.time()
frame_count = 0

print("Iniciando prueba de detección facial...")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: No se pudo capturar el frame")
        break
    
    frame_count += 1
    tiempo_actual = time.time() - tiempo_inicio
    
    # Si ha pasado el tiempo de prueba, terminar
    if tiempo_actual > DURACION_PRUEBA:
        print(f"Prueba completada. Procesados {frame_count} frames en {tiempo_actual:.2f} segundos")
        break
    
    # Procesar frame con todos los métodos
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- HOG ---
    hog_start = time.time()
    hog_rects, _ = detector_hog.detectMultiScale(frame_gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    hog_time = time.time() - hog_start
    
    # --- HAAR ---
    haar_start = time.time()
    haar_rects = detector_haar.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=4)
    haar_time = time.time() - haar_start
    
    # --- MTCNN ---
    mtcnn_start = time.time()
    mtcnn_results = detector_mtcnn.detect_faces(frame_rgb)
    mtcnn_rects = []
    for res in mtcnn_results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        mtcnn_rects.append([x, y, w, h])
    mtcnn_time = time.time() - mtcnn_start
    
    # Almacenar resultados
    resultados.append({
        "frame": frame_count,
        "tiempo": tiempo_actual,
        "HOG_detecciones": len(hog_rects),
        "HOG_tiempo": hog_time,
        "HAAR_detecciones": len(haar_rects),
        "HAAR_tiempo": haar_time,
        "MTCNN_detecciones": len(mtcnn_rects),
        "MTCNN_tiempo": mtcnn_time
    })
    
    # Reconocimiento facial usando MTCNN (el más preciso)
    for res in mtcnn_results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        
        # Extraer región facial
        face_region = frame_rgb[y:y+h, x:x+w]
        
        # Inside your video processing loop
        face_locations = face_recognition.face_locations(frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Get encoding using FULL FRAME + LOCATION
            face_encodings = face_recognition.face_encodings(
                frame, 
                known_face_locations=[face_location],
                num_jitters=1
            )
            
            if not face_encodings:
                continue  # Skip if no encoding found
                
            face_encoding = face_encodings[0]
            
    # Mostrar información de rendimiento
    cv2.putText(frame, f"Tiempo: {tiempo_actual:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"HOG: {len(hog_rects)} rostros", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"HAAR: {len(haar_rects)} rostros", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(frame, f"MTCNN: {len(mtcnn_rects)} rostros", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.imshow('Comparacion Modelos', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()

# Procesar resultados y generar gráficos
if len(resultados) > 0:
    # Guardar CSV con resultados
    csv_path = os.path.join(RESULTADOS_DIR, "metricas_video.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        campos = ["frame", "tiempo", "HOG_detecciones", "HOG_tiempo", 
                 "HAAR_detecciones", "HAAR_tiempo", "MTCNN_detecciones", "MTCNN_tiempo"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    
    # Preparar datos para gráficos
    frames = [r['frame'] for r in resultados]
    hog_dets = [r['HOG_detecciones'] for r in resultados]
    haar_dets = [r['HAAR_detecciones'] for r in resultados]
    mtcnn_dets = [r['MTCNN_detecciones'] for r in resultados]
    
    hog_times = [r['HOG_tiempo'] for r in resultados]
    haar_times = [r['HAAR_tiempo'] for r in resultados]
    mtcnn_times = [r['MTCNN_tiempo'] for r in resultados]
    
    # Gráfico 1: Detecciones por frame
    plt.figure(figsize=(12, 6))
    plt.plot(frames, hog_dets, 'g-', label='HOG', alpha=0.7)
    plt.plot(frames, haar_dets, 'r-', label='HAAR', alpha=0.7)
    plt.plot(frames, mtcnn_dets, 'y-', label='MTCNN', alpha=0.7)
    plt.title('Detecciones por Frame')
    plt.xlabel('Número de Frame')
    plt.ylabel('Detecciones')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTADOS_DIR, 'detecciones_por_frame.png'), dpi=150)
    
    # Gráfico 2: Tiempo de procesamiento por frame
    plt.figure(figsize=(12, 6))
    plt.plot(frames, hog_times, 'g-', label='HOG', alpha=0.7)
    plt.plot(frames, haar_times, 'r-', label='HAAR', alpha=0.7)
    plt.plot(frames, mtcnn_times, 'y-', label='MTCNN', alpha=0.7)
    plt.title('Tiempo de Procesamiento por Frame')
    plt.xlabel('Número de Frame')
    plt.ylabel('Tiempo (s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTADOS_DIR, 'tiempos_por_frame.png'), dpi=150)
    
    # Gráfico 3: Comparación de métricas promedio
    metodos = ['HOG', 'HAAR', 'MTCNN']
    detecciones_promedio = [np.mean(hog_dets), np.mean(haar_dets), np.mean(mtcnn_dets)]
    tiempo_promedio = [np.mean(hog_times), np.mean(haar_times), np.mean(mtcnn_times)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(metodos, detecciones_promedio, color=['green', 'red', 'yellow'])
    ax1.set_title('Detecciones Promedio por Frame')
    ax1.set_ylabel('Número de Detecciones')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(metodos, tiempo_promedio, color=['green', 'red', 'yellow'])
    ax2.set_title('Tiempo Promedio de Procesamiento')
    ax2.set_ylabel('Tiempo (s)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS_DIR, 'comparacion_metodos.png'), dpi=150)
    
    print(f"Análisis completado. Resultados guardados en {RESULTADOS_DIR}")
else:
    print("No se obtuvieron resultados para analizar")