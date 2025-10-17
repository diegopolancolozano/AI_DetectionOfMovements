"""
Script para enriquecer el dataset con features derivados:
- Velocidades por landmark clave
- Ángulos de articulaciones (rodillas, codos)
- Segmentación temporal por etiqueta
- Marca de baja calidad
"""

import math

import numpy as np
import pandas as pd

SRC = "mediapipe_labels_dataset.csv"
DST = "mediapipe_labels_dataset_enriched.csv"

print(f"📂 Cargando: {SRC}")
df = pd.read_csv(SRC).sort_values(["video_id", "frame_opencv"]).reset_index(drop=True)

print(f"   ✓ {len(df)} frames cargados")

# Asegurar fps válido por video
df["fps_eff"] = df.groupby("video_id")["fps"].transform(
    lambda s: s.fillna(s.median()).replace(0, s.median()).fillna(30)
)

# Velocidades para landmarks clave
# 15/16: muñecas, 25/26: rodillas, 27/28: tobillos
keys = [15, 16, 25, 26, 27, 28]
print(f"📊 Calculando velocidades para landmarks: {keys}")
for i in keys:
    dx = df.groupby("video_id")[f"x_{i}"].diff()
    dy = df.groupby("video_id")[f"y_{i}"].diff()
    df[f"speed_{i}"] = np.sqrt(dx.fillna(0)**2 + dy.fillna(0)**2) * df["fps_eff"]

# Función para calcular ángulo (A-B-C en grados)
def angle_deg(ax, ay, bx, by, cx, cy):
    """
    Calcula el ángulo en B del triángulo A-B-C.
    Retorna ángulo en grados [0, 180].
    """
    BAx, BAy = ax - bx, ay - by
    BCx, BCy = cx - bx, cy - by
    num = BAx * BCx + BAy * BCy
    den = np.sqrt(BAx**2 + BAy**2) * np.sqrt(BCx**2 + BCy**2)
    # Evitar división por cero y asegurar cosang en [-1, 1]
    cosang = np.clip(num / np.where(den == 0, np.nan, den), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# Ángulos de articulaciones clave
print("📐 Calculando ángulos de articulaciones...")

# Rodillas: cadera(23/24) - rodilla(25/26) - tobillo(27/28)
df["knee_left_deg"] = angle_deg(
    df["x_23"], df["y_23"],  # cadera izq
    df["x_25"], df["y_25"],  # rodilla izq
    df["x_27"], df["y_27"]   # tobillo izq
)
df["knee_right_deg"] = angle_deg(
    df["x_24"], df["y_24"],  # cadera der
    df["x_26"], df["y_26"],  # rodilla der
    df["x_28"], df["y_28"]   # tobillo der
)

# Codos: hombro(11/12) - codo(13/14) - muñeca(15/16)
df["elbow_left_deg"] = angle_deg(
    df["x_11"], df["y_11"],  # hombro izq
    df["x_13"], df["y_13"],  # codo izq
    df["x_15"], df["y_15"]   # muñeca izq
)
df["elbow_right_deg"] = angle_deg(
    df["x_12"], df["y_12"],  # hombro der
    df["x_14"], df["y_14"],  # codo der
    df["x_16"], df["y_16"]   # muñeca der
)

# Segmentos contiguos por etiqueta (para análisis temporal)
print("📍 Creando segmentos por etiqueta...")
df["segment_id"] = (
    df["label"].ne(df.groupby("video_id")["label"].shift())
    .groupby(df["video_id"]).cumsum()
)

# Marca de baja calidad
if "mean_visibility" in df and "num_visible_lms" in df:
    df["low_quality"] = (df["mean_visibility"] < 0.5) | (df["num_visible_lms"] < 15)
    n_low = df["low_quality"].sum()
    print(f"⚠️  Frames de baja calidad: {n_low} ({100*n_low/len(df):.1f}%)")

print(f"💾 Guardando: {DST}")
df.to_csv(DST, index=False)

print(f"✅ Enriquecimiento completado")
print(f"   Nuevas columnas:")
print(f"   - Metadatos: fps, timestamp_ms, width, height")
print(f"   - Calidad: mean_visibility, num_visible_lms, low_quality")
print(f"   - Posición/escala: hip_center_x, hip_center_y, torso_scale")
print(f"   - Bounding box: bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_area, bbox_aspect")
print(f"   - Velocidades: speed_15, speed_16, speed_25, speed_26, speed_27, speed_28")
print(f"   - Ángulos: knee_left_deg, knee_right_deg, elbow_left_deg, elbow_right_deg")
print(f"   - Segmentación: segment_id")
print(f"   Shape final: {df.shape}")
