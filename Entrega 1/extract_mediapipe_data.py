import json
import math
import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
VIDEOS_DIR = "Videos APO"
LABEL_FILE = "project-label-studio.json"
OUTPUT_CSV = "mediapipe_labels_dataset.csv"

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

def process_video(video_path, video_id):
    """Extrae landmarks de cada frame y los une con etiquetas temporales."""
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
    # OpenCV cuenta todos los frames, Label Studio puede usar un índice diferente
    frame_ratio = (max_frame_labelstudio / total_frames_opencv) if (max_frame_labelstudio and total_frames_opencv) else 1.0
    
    print(f"  📊 Frames OpenCV: {total_frames_opencv}, Label Studio máx: {max_frame_labelstudio}, FPS: {fps:.2f}")
    print(f"  🔄 Ratio de conversión: {frame_ratio:.4f}")
    
    frame_idx_opencv = 0
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
            
            row = {
                'video_id': video_id, 
                'frame_opencv': frame_idx_opencv,
                'frame_labelstudio': frame_idx_labelstudio,
                'fps': fps,
                'timestamp_ms': (frame_idx_opencv / fps * 1000.0) if fps > 0 else None,
                'width': width,
                'height': height
            }
            
            xs, ys, vis_vals = [], [], []
            for i, lm in enumerate(landmarks):
                row[f'x_{i}'] = lm.x
                row[f'y_{i}'] = lm.y
                row[f'z_{i}'] = lm.z
                row[f'v_{i}'] = lm.visibility
                xs.append(lm.x)
                ys.append(lm.y)
                vis_vals.append(lm.visibility)

            # Calidad: media de visibility y número de landmarks visibles
            row['mean_visibility'] = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0
            row['num_visible_lms'] = int(sum(1 for v in vis_vals if v >= 0.5))

            # Centro de caderas y escala de torso (normalización espacial)
            try:
                lh, rh = landmarks[23], landmarks[24]   # left/right hip
                ls, rs = landmarks[11], landmarks[12]   # left/right shoulder
                hip_cx = (lh.x + rh.x) / 2.0
                hip_cy = (lh.y + rh.y) / 2.0
                # Distancias de torso en coordenadas normalizadas
                d_l = math.hypot(ls.x - lh.x, ls.y - lh.y)
                d_r = math.hypot(rs.x - rh.x, rs.y - rh.y)
                torso_scale = max((d_l + d_r) / 2.0, 1e-6)
                row['hip_center_x'] = hip_cx
                row['hip_center_y'] = hip_cy
                row['torso_scale'] = torso_scale
            except Exception:
                row['hip_center_x'] = None
                row['hip_center_y'] = None
                row['torso_scale'] = None

            # Bounding box del esqueleto
            if xs and ys:
                xmin, xmax = float(min(xs)), float(max(xs))
                ymin, ymax = float(min(ys)), float(max(ys))
                row['bbox_xmin'] = xmin
                row['bbox_ymin'] = ymin
                row['bbox_xmax'] = xmax
                row['bbox_ymax'] = ymax
                row['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
                row['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) not in (0, None) else None
            else:
                row['bbox_xmin'] = row['bbox_ymin'] = row['bbox_xmax'] = row['bbox_ymax'] = None
                row['bbox_area'] = row['bbox_aspect'] = None

            row['label'] = extract_label_for_frame(video_id, frame_idx_labelstudio)
            data.append(row)

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
    base_name = json_video_name.split("-")[-1]  # ej: Video_1.mp4, Video_12.mp4
    
    # Normalizar el nombre (remover guión bajo y agregar espacio)
    # Video_1.mp4 -> Video 1.mp4
    video_num = base_name.replace("Video_", "").replace(".mp4", "")
    real_video_name = f"Video {video_num}.mp4"
    
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
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset guardado en: {OUTPUT_CSV}")
else:
    print("⚠️ No se generó ningún dataset.")
