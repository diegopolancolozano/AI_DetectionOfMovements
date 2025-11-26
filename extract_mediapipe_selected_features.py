"""
Script para extraer SOLO las features especificadas del dataset de MediaPipe.
Extrae landmarks de video y calcula únicamente las 18 características seleccionadas + label.
"""

import json
import math
import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
VIDEOS_DIR = "Entrega 2/videos familiaDiego"
LABEL_FILE = "Entrega 2/videos familiaDiego/project-4-at-2025-11-02-12-56-089d6ad2.json"
OUTPUT_CSV = "data/v1/mediapipe_labels_dataset_selected_features_pola.csv"

# Features que queremos extraer
FEATURE_NAMES = [
    'mean_visibility',
    'num_visible_lms',
    'hip_center_x',
    'hip_center_y',
    'torso_scale',
    'bbox_area',
    'bbox_aspect',
    'knee_left_deg',
    'knee_right_deg',
    'elbow_left_deg',
    'elbow_right_deg',
    'speed_15',   # velocidad muñeca izq
    'speed_16',   # velocidad muñeca der
    'velocidad_centro_cuerpo',
    'apertura_piernas_norm',
    'ancho_hombros',
    'altura_normalizada',
    'dist_vertical_cabeza_cadera'
]

# === MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# === CARGAR ETIQUETAS ===
with open(LABEL_FILE, 'r') as f:
    label_data = json.load(f)

def get_max_label_frame(video_id):
    """Obtiene el frame máximo etiquetado en Label Studio para un video."""
    for entry in label_data:
        if entry["id"] == video_id:
            results = entry["annotations"][0]["result"]
            max_frame = 0
            for r in results:
                end = r["value"]["ranges"][0]["end"]
                if end > max_frame:
                    max_frame = end
            return max_frame
    return None

def extract_label_for_frame(video_id, frame_idx_labelstudio):
    """Devuelve la etiqueta (actividad) correspondiente a un frame según el JSON."""
    for entry in label_data:
        if entry["id"] == video_id:
            results = entry["annotations"][0]["result"]
            for r in results:
                start = r["value"]["ranges"][0]["start"]
                end = r["value"]["ranges"][0]["end"]
                label = r["value"]["timelinelabels"][0]
                if start <= frame_idx_labelstudio <= end:
                    return label
    return "Unlabeled"

def angle_deg(ax, ay, bx, by, cx, cy):
    """
    Calcula el ángulo en B del triángulo A-B-C.
    Retorna ángulo en grados [0, 180].
    """
    BAx, BAy = ax - bx, ay - by
    BCx, BCy = cx - bx, cy - by
    num = BAx * BCx + BAy * BCy
    den = math.sqrt(BAx**2 + BAy**2) * math.sqrt(BCx**2 + BCy**2)
    if den == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, num / den))
    return math.degrees(math.acos(cosang))

def process_video(video_path, video_id):
    """Extrae landmarks de cada frame y calcula solo las features seleccionadas."""
    cap = cv2.VideoCapture(video_path)
    data = []

    # Obtener información del video
    total_frames_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Obtener el frame máximo de Label Studio
    max_frame_labelstudio = get_max_label_frame(video_id)
    
    # Calcular factor de conversión
    frame_ratio = (max_frame_labelstudio / total_frames_opencv) if (max_frame_labelstudio and total_frames_opencv) else 1.0
    
    print(f"  📊 Frames OpenCV: {total_frames_opencv}, Label Studio máx: {max_frame_labelstudio}, FPS: {fps:.2f}")
    print(f"  🔄 Ratio de conversión: {frame_ratio:.4f}")
    
    frame_idx_opencv = 0
    prev_landmarks = None  # Para calcular velocidades
    prev_hip_cx, prev_hip_cy = None, None  # Para velocidad del centro
    
    pbar = tqdm(total=total_frames_opencv, desc=os.path.basename(video_path))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Convertir el frame de OpenCV al índice de Label Studio
            frame_idx_labelstudio = int(frame_idx_opencv * frame_ratio)
            
            # Diccionario para almacenar solo las features seleccionadas
            row = {
                'video_id': video_id, 
                'frame_opencv': frame_idx_opencv,
                'frame_labelstudio': frame_idx_labelstudio,
            }
            
            # === EXTRAER COORDENADAS Y VISIBILIDAD ===
            xs, ys, vis_vals = [], [], []
            for lm in landmarks:
                xs.append(lm.x)
                ys.append(lm.y)
                vis_vals.append(lm.visibility)

            # === 1. mean_visibility ===
            row['mean_visibility'] = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0
            
            # === 2. num_visible_lms ===
            row['num_visible_lms'] = int(sum(1 for v in vis_vals if v >= 0.5))

            # === 3-5. hip_center_x, hip_center_y, torso_scale ===
            try:
                lh, rh = landmarks[23], landmarks[24]   # left/right hip
                ls, rs = landmarks[11], landmarks[12]   # left/right shoulder
                hip_cx = (lh.x + rh.x) / 2.0
                hip_cy = (lh.y + rh.y) / 2.0
                d_l = math.hypot(ls.x - lh.x, ls.y - lh.y)
                d_r = math.hypot(rs.x - rh.x, rs.y - rh.y)
                torso_scale = max((d_l + d_r) / 2.0, 1e-6)
                row['hip_center_x'] = hip_cx
                row['hip_center_y'] = hip_cy
                row['torso_scale'] = torso_scale
            except Exception:
                hip_cx, hip_cy = None, None
                torso_scale = None
                row['hip_center_x'] = None
                row['hip_center_y'] = None
                row['torso_scale'] = None

            # === 6-7. bbox_area, bbox_aspect ===
            if xs and ys:
                xmin, xmax = float(min(xs)), float(max(xs))
                ymin, ymax = float(min(ys)), float(max(ys))
                row['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
                row['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else None
            else:
                row['bbox_area'] = None
                row['bbox_aspect'] = None

            # === 8-9. knee_left_deg, knee_right_deg ===
            row['knee_left_deg'] = angle_deg(
                landmarks[23].x, landmarks[23].y,  # cadera izq
                landmarks[25].x, landmarks[25].y,  # rodilla izq
                landmarks[27].x, landmarks[27].y   # tobillo izq
            )
            row['knee_right_deg'] = angle_deg(
                landmarks[24].x, landmarks[24].y,  # cadera der
                landmarks[26].x, landmarks[26].y,  # rodilla der
                landmarks[28].x, landmarks[28].y   # tobillo der
            )

            # === 10-11. elbow_left_deg, elbow_right_deg ===
            row['elbow_left_deg'] = angle_deg(
                landmarks[11].x, landmarks[11].y,  # hombro izq
                landmarks[13].x, landmarks[13].y,  # codo izq
                landmarks[15].x, landmarks[15].y   # muñeca izq
            )
            row['elbow_right_deg'] = angle_deg(
                landmarks[12].x, landmarks[12].y,  # hombro der
                landmarks[14].x, landmarks[14].y,  # codo der
                landmarks[16].x, landmarks[16].y   # muñeca der
            )

            # === 12-13. speed_15, speed_16 (velocidad muñecas) ===
            if prev_landmarks is not None and fps > 0:
                dx_15 = landmarks[15].x - prev_landmarks[15].x
                dy_15 = landmarks[15].y - prev_landmarks[15].y
                row['speed_15'] = math.sqrt(dx_15**2 + dy_15**2) * fps
                
                dx_16 = landmarks[16].x - prev_landmarks[16].x
                dy_16 = landmarks[16].y - prev_landmarks[16].y
                row['speed_16'] = math.sqrt(dx_16**2 + dy_16**2) * fps
            else:
                row['speed_15'] = 0.0
                row['speed_16'] = 0.0

            # === 14. velocidad_centro_cuerpo ===
            if prev_hip_cx is not None and prev_hip_cy is not None and hip_cx is not None and fps > 0:
                dx_center = hip_cx - prev_hip_cx
                dy_center = hip_cy - prev_hip_cy
                row['velocidad_centro_cuerpo'] = math.sqrt(dx_center**2 + dy_center**2) * fps
            else:
                row['velocidad_centro_cuerpo'] = 0.0

            # === 15. apertura_piernas_norm ===
            if torso_scale and torso_scale > 0:
                apertura_piernas = abs(landmarks[27].x - landmarks[28].x)  # tobillos
                row['apertura_piernas_norm'] = apertura_piernas / torso_scale
            else:
                row['apertura_piernas_norm'] = None

            # === 16. ancho_hombros ===
            row['ancho_hombros'] = abs(landmarks[11].x - landmarks[12].x)

            # === 17. altura_normalizada ===
            try:
                nose = landmarks[0]
                ankle_l, ankle_r = landmarks[27], landmarks[28]
                ankle_avg_y = (ankle_l.y + ankle_r.y) / 2.0
                row['altura_normalizada'] = abs(nose.y - ankle_avg_y)
            except Exception:
                row['altura_normalizada'] = None

            # === 18. dist_vertical_cabeza_cadera ===
            try:
                nose = landmarks[0]
                hip_avg_y = (landmarks[23].y + landmarks[24].y) / 2.0
                row['dist_vertical_cabeza_cadera'] = abs(nose.y - hip_avg_y)
            except Exception:
                row['dist_vertical_cabeza_cadera'] = None

            # === LABEL ===
            row['label'] = extract_label_for_frame(video_id, frame_idx_labelstudio)
            
            data.append(row)
            
            # Guardar landmarks y posición del centro para el siguiente frame
            prev_landmarks = landmarks
            prev_hip_cx, prev_hip_cy = hip_cx, hip_cy

        frame_idx_opencv += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return pd.DataFrame(data)

# === PROCESAR TODOS LOS VIDEOS ===
all_data = []

# Crear un mapeo directo entre el nombre del video en el JSON y el archivo real
video_mapping = {}
for entry in label_data:
    json_video_name = entry["file_upload"].split("/")[-1]
    # Extraer el nombre real del video (después del hash/UUID)
    # ej: "7a2d1d80-vid1.mp4" -> "vid1.mp4"
    real_video_name = json_video_name.split("-", 1)[-1] if "-" in json_video_name else json_video_name
    
    video_mapping[entry["id"]] = real_video_name

# Verificar qué videos existen en el directorio
available_videos = os.listdir(VIDEOS_DIR)
print("Videos disponibles en el directorio:")
for video in sorted(available_videos):
    print(f"  - {video}")

print("\n" + "="*60)
print("MAPEO DE VIDEOS:")
print("="*60)
for json_id, expected_name in sorted(video_mapping.items()):
    status = "✅" if expected_name in available_videos else "❌"
    print(f"{status} ID {json_id:2d} -> {expected_name}")
print("="*60 + "\n")

# Procesar cada video según el mapeo
for entry in label_data:
    video_id = entry["id"]
    expected_video_name = video_mapping[video_id]
    
    if expected_video_name in available_videos:
        video_path = os.path.join(VIDEOS_DIR, expected_video_name)
        print(f"\n✅ Procesando {expected_video_name} (ID {video_id}) ...")
        df = process_video(video_path, video_id)
        all_data.append(df)
    else:
        print(f"\n❌ [ERROR] No se encontró el archivo: {expected_video_name} (ID {video_id})")

# === GUARDAR DATASET ===
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Verificar que todas las columnas esperadas estén presentes
    expected_cols = ['video_id', 'frame_opencv', 'frame_labelstudio'] + FEATURE_NAMES + ['label']
    actual_cols = list(final_df.columns)
    
    print(f"\n📋 Columnas generadas: {len(actual_cols)}")
    print(f"   Features: {FEATURE_NAMES}")
    print(f"   + Metadatos: video_id, frame_opencv, frame_labelstudio")
    print(f"   + Label: label")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset guardado en: {OUTPUT_CSV}")
    print(f"   Shape: {final_df.shape}")
    print(f"\n📊 Distribución de labels:")
    print(final_df['label'].value_counts())
else:
    print("⚠️ No se generó ningún dataset.")
