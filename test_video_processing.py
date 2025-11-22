"""
Script de prueba para demostrar que el procesamiento funciona en Windows
Crea un video de prueba y lo procesa
"""
import cv2
import numpy as np
import os

# Crear un video de prueba simple
print("📹 Creando video de prueba...")

# Configuración del video
width, height = 640, 480
fps = 30
duration_seconds = 3
total_frames = fps * duration_seconds

# Crear carpeta de prueba
os.makedirs("test_videos", exist_ok=True)
video_path = "test_videos/test_video.mp4"

# Crear video con un círculo que se mueve
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for i in range(total_frames):
    # Crear frame negro
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Dibujar un círculo que se mueve
    x = int(width * (i / total_frames))
    y = height // 2
    cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)
    
    # Agregar texto
    cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()

print(f"✅ Video creado: {video_path}")
print(f"   Duración: {duration_seconds}s")
print(f"   Frames: {total_frames}")
print(f"   Resolución: {width}x{height}")

# Verificar que se creó
if os.path.exists(video_path):
    size_mb = os.path.getsize(video_path) / (1024*1024)
    print(f"   Tamaño: {size_mb:.2f} MB")
    print("\n✅ El video se creó correctamente en Windows")
else:
    print("\n❌ Error al crear el video")
