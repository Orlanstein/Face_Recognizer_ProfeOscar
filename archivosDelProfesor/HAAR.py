#!/usr/bin/env python3
import cv2
import time
import numpy as np
import psutil
import os

class MFNFaceTracker:
    def __init__(self):
        # Configuración de cámara
        self.camera_index = self.detect_camera()
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Usar Haar como detector inicial
        cascade_path = os.path.join(os.path.dirname(__file__), '/home/orin/Downloads/pony/config/haarcascades/haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            print(f"Error: No se encontró el archivo Haar cascade en: {cascade_path}")
            return
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Error: No se pudo cargar el clasificador Haar cascade")
            return
        
        self.min_face_size = 80
        
        # Variables de rendimiento
        self.frame_count = 0
        self.fps = 0
        self.cpu_usage = 0
        self.mem_usage = 0
        self.face_count = 0
        self.last_time = time.time()
        
        print("HAAR Tracker iniciado")

    def detect_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return 0

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size))
        
        # Convertir de (x,y,w,h) a (top,right,bottom,left)
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
        
        return face_locations

    def draw_metrics(self, frame):
        # Configuración de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (255, 0, 255)  # Magenta
        y_offset = 30
        line_height = 30
        
        # Texto a mostrar
        metrics = [
            f"FPS: {self.fps:.1f}",
            f"CPU: {self.cpu_usage:.1f}%",
            f"MEM: {self.mem_usage:.1f}%",
            f"Faces: {self.face_count}"
        ]
        
        # Dibujar cada métrica
        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, y_offset + i * line_height), 
                       font, font_scale, color, font_thickness, cv2.LINE_AA)

    def run(self):
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Advertencia: Error al capturar frame")
                    continue
                
                face_locations = self.detect_faces(frame)
                self.face_count = len(face_locations)
                
                # Calcular métricas
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
                    
                    # Obtener uso de recursos
                    self.cpu_usage = psutil.cpu_percent()
                    self.mem_usage = psutil.virtual_memory().percent
                    
                    # Mostrar métricas en consola
                    print(f"MFN|FPS:{self.fps:.1f}|Faces:{self.face_count}|CPU:{self.cpu_usage:.1f}|MEM:{self.mem_usage:.1f}")
                
                # Mostrar resultados
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                
                # Dibujar métricas en el frame
                self.draw_metrics(frame)
                
                cv2.imshow('Face Tracking HAAR', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = MFNFaceTracker()
    tracker.run()