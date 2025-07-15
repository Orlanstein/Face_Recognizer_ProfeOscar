# comparacion_mtcnn_hog_haar.py
#
# Se uso el siguiente dataset:
# https://universe.roboflow.com/fddb/face-detection-40nq0/dataset/1
#

import cv2
import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from mtcnn import MTCNN

# Crear carpeta para guardar resultados
os.makedirs("resultados", exist_ok=True)

# Cargar imagenes de prueba con manejo de errores
image_dir = "imagenes_prueba"
imagenes = []
for f in os.listdir(image_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, f)
        img = cv2.imread(img_path)
        if img is not None:
            imagenes.append(img)
        else:
            print(f"Advertencia: No se pudo cargar {img_path}")

if not imagenes:
    print("Error: No se encontraron imágenes válidas en el directorio.")
    exit()

# --- HOG (Personas) ---
def reconocimiento_hog(imagen):
    start = time.time()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(imagen_gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    tiempo = time.time() - start
    return rects, tiempo

# --- HAAR (Rostros) ---
def reconocimiento_haar(imagen):
    start = time.time()
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.isfile(cascade_path):
        print(f"Error: No se encontró el archivo Haar cascade: {cascade_path}")
        return [], 0
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imagen_gray, scaleFactor=1.1, minNeighbors=4)
    tiempo = time.time() - start
    return faces, tiempo

# --- MTCNN (Rostros) ---
def reconocimiento_mtcnn(imagen):
    start = time.time()
    detector = MTCNN()
    
    # Convertir BGR a RGB (MTCNN espera imágenes RGB)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # Detectar rostros
    resultados = detector.detect_faces(imagen_rgb)
    
    rects = []
    for resultado in resultados:
        x, y, w, h = resultado['box']
        # Asegurar que las coordenadas no sean negativas
        x, y = max(0, x), max(0, y)
        w, h = max(0, w), max(0, h)
        rects.append([x, y, w, h])
    
    tiempo = time.time() - start
    return rects, tiempo

# Ejecutar comparación y guardar resultados
resultados = []

# Crear carpeta para visualizaciones
# os.makedirs("resultados/visualizaciones", exist_ok=True)

# Colores para cada método
colores_metodos = {
    "HOG": (0, 255, 0),      # Verde
    "HAAR": (0, 0, 255),     # Rojo
    "MTCNN": (255, 255, 0)   # Amarillo
}

for idx, img in enumerate(imagenes):
    print(f"\nProcesando imagen {idx+1}/{len(imagenes)}")
    img_visual = img.copy()
    
    # Procesar con todos los métodos
    for metodo, funcion in [
        ("HOG", reconocimiento_hog),
        ("HAAR", reconocimiento_haar),
        ("MTCNN", reconocimiento_mtcnn)
    ]:
        try:
            detecciones, tiempo = funcion(img)
            print(f"{metodo}: {len(detecciones)} detecciones en {tiempo:.4f}s")
            
            # Guardar resultados
            resultados.append({
                "imagen": f"imagen_{idx+1}",
                "metodo": metodo,
                "detecciones": len(detecciones),
                "tiempo": tiempo
            })
            
            # Crear imagen con las detecciones de este método
            img_metodo = img.copy()
            color = colores_metodos[metodo]
            
            for det in detecciones:
                # Todos los métodos devuelven [x, y, w, h]
                x, y, w, h = det
                cv2.rectangle(img_metodo, (x, y), (x+w, y+h), color, 2)
            
            # Guardar imagen individual
            # cv2.imwrite(f"resultados/visualizaciones/imagen_{idx+1}_{metodo}.jpg", img_metodo)
            
            # Añadir a la imagen compuesta
            for det in detecciones:
                x, y, w, h = det
                cv2.rectangle(img_visual, (x, y), (x+w, y+h), color, 2)
                # Etiqueta del método
                cv2.putText(img_visual, metodo, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            print(f"Error con {metodo}: {str(e)}")

    # Tope artificial, para no usar todas las imagenes del dataset
    if idx >= 100:
        break
    
    # Guardar imagen compuesta con todas las detecciones
    # cv2.imwrite(f"resultados/visualizaciones/imagen_{idx+1}_TODOS.jpg", img_visual)

# Guardar CSV
csv_path = "resultados/metricas.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["imagen", "metodo", "detecciones", "tiempo"])
    writer.writeheader()
    writer.writerows(resultados)
    print(f"\nResultados guardados en {csv_path}")

# Procesar métricas para gráficos
if not resultados:
    print("No hay resultados para graficar")
    exit()

metodos = ["HOG", "HAAR", "MTCNN"]
colores = ['#1f77b4', '#ff7f0e', '#d62728']

# Tiempo promedio
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
tiempo_promedio = []
for metodo in metodos:
    tiempos = [r['tiempo'] for r in resultados if r['metodo'] == metodo]
    if tiempos:
        avg_time = np.mean(tiempos)
        tiempo_promedio.append(avg_time)
    else:
        tiempo_promedio.append(0)

plt.bar(metodos, tiempo_promedio, color=colores)
plt.title("Tiempo promedio por método (s)")
plt.ylabel("Tiempo (s)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Total de detecciones
plt.subplot(2, 2, 2)
detecciones_totales = []
for metodo in metodos:
    dets = [r['detecciones'] for r in resultados if r['metodo'] == metodo]
    detecciones_totales.append(np.sum(dets))

plt.bar(metodos, detecciones_totales, color=colores)
plt.title("Total de detecciones por método")
plt.ylabel("Número de detecciones")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Gráfico de dispersión: Tiempo vs Detecciones
plt.subplot(2, 2, 3)
for i, metodo in enumerate(metodos):
    tiempos = [r['tiempo'] for r in resultados if r['metodo'] == metodo]
    dets = [r['detecciones'] for r in resultados if r['metodo'] == metodo]
    plt.scatter(tiempos, dets, color=colores[i], s=100, alpha=0.7, label=metodo)

plt.title("Relación Tiempo vs Detecciones")
plt.xlabel("Tiempo de procesamiento (s)")
plt.ylabel("Número de detecciones")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfico de caja para tiempos de ejecución
plt.subplot(2, 2, 4)
datos_tiempos = []
for metodo in metodos:
    tiempos = [r['tiempo'] for r in resultados if r['metodo'] == metodo]
    datos_tiempos.append(tiempos)

plt.boxplot(datos_tiempos, labels=metodos, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='darkblue'),
            medianprops=dict(color='red'))
plt.title("Distribución de Tiempos por Método")
plt.ylabel("Tiempo (s)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("resultados/comparacion_metricas.png", dpi=150)
print("Gráficos guardados en resultados/comparacion_metricas.png")

# Gráfico de precisión relativa (asumiendo MTCNN como referencia)
if len(imagenes) > 0:
    plt.figure(figsize=(10, 6))
    precision_relativa = {"HOG": [], "HAAR": []}
    
    for i in range(len(imagenes)):
        dets_mtcnn = [r['detecciones'] for r in resultados 
                   if r['imagen'] == f"imagen_{i+1}" and r['metodo'] == "MTCNN"]
        
        if not dets_mtcnn:
            continue
            
        ref_detections = dets_mtcnn[0]
        
        for metodo in ["HOG", "HAAR"]:
            dets = [r['detecciones'] for r in resultados 
                   if r['imagen'] == f"imagen_{i+1}" and r['metodo'] == metodo]
            
            if dets:
                ratio = dets[0] / ref_detections if ref_detections > 0 else 0
                precision_relativa[metodo].append(ratio)
    
    # Calcular precisión media
    for metodo, ratios in precision_relativa.items():
        if ratios:
            plt.plot(ratios, 'o-', label=f"{metodo} vs MTCNN", alpha=0.7)
    
    plt.axhline(y=1, color='r', linestyle='--', label='Referencia MTCNN')
    plt.title("Precisión Relativa (vs MTCNN)")
    plt.xlabel("Número de Imagen")
    plt.ylabel("Ratio de Detección (Método / MTCNN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("resultados/precision_relativa.png", dpi=150)
    print("Gráfico de precisión relativa guardado en resultados/precision_relativa.png")

print("Proceso completado exitosamente")