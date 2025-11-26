"""
Script para procesar TODOS los videos del proyecto (Entrega 1 y Entrega 2)
y combinar los datos en un unico archivo CSV para analisis y modelado.

Este script:
1. Procesa videos de Entrega 1 (Videos APO)
2. Procesa videos de Entrega 2/videos familiaDiego
3. Procesa videos de Entrega 2/videos otroGrupo
4. Combina todos los datos en: mediapipe_labels_dataset_combined.csv
5. Aplica enriquecimiento de features
"""

import json
import math
import os
from pathlib import Path

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIGURACION ===
# Obtener directorio del script actual
SCRIPT_DIR = Path(__file__).parent.absolute()

ENTREGA1_VIDEOS_DIR = SCRIPT_DIR / ".." / "Entrega 1" / "Videos APO" / "Videos APO"
ENTREGA1_LABEL_FILE = SCRIPT_DIR / ".." / "Entrega 1" / "project-label-studio.json"

ENTREGA2_FAMILIA_VIDEOS_DIR = SCRIPT_DIR / "videos familiaDiego"
ENTREGA2_FAMILIA_LABEL_FILE = SCRIPT_DIR / "videos familiaDiego" / "project-4-at-2025-11-02-12-56-089d6ad2.json"

ENTREGA2_OTRO_VIDEOS_DIR = SCRIPT_DIR / "videos otroGrupo"
ENTREGA2_OTRO_LABEL_FILE = SCRIPT_DIR / "videos otroGrupo" / "project-5-at-2025-11-02-14-50-27ad6e06.json"

OUTPUT_CSV = SCRIPT_DIR / "mediapipe_labels_dataset_combined.csv"
OUTPUT_ENRICHED_CSV = SCRIPT_DIR / "mediapipe_labels_dataset_combined_enriched.csv"

# === MEDIAPIPE ===
print("Inicializando MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# === FUNCIONES AUXILIARES ===

def load_label_data(label_file):
    """Carga el archivo JSON de Label Studio"""
    label_file = Path(label_file)
    if not label_file.exists():
        print(f"Advertencia: No se encontro: {label_file}")
        return []
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_max_label_frame(video_id, label_data):
    """Obtiene el frame maximo etiquetado en Label Studio para un video."""
    for entry in label_data:
        if entry["id"] == video_id:
            if "annotations" not in entry or not entry["annotations"]:
                return None
            results = entry["annotations"][0]["result"]
            max_frame = 0
            for r in results:
                end = r["value"]["ranges"][0]["end"]
                if end > max_frame:
                    max_frame = end
            return max_frame
    return None

def extract_label_for_frame(video_id, frame_idx_labelstudio, label_data):
    """Devuelve la etiqueta (actividad) correspondiente a un frame segun el JSON."""
    for entry in label_data:
        if entry["id"] == video_id:
            if "annotations" not in entry or not entry["annotations"]:
                return "Unlabeled"
            results = entry["annotations"][0]["result"]
            for r in results:
                start = r["value"]["ranges"][0]["start"]
                end = r["value"]["ranges"][0]["end"]
                label = r["value"]["timelinelabels"][0]
                if start <= frame_idx_labelstudio <= end:
                    return label
    return "Unlabeled"

def combine_classes(label):
    """Combina clases similares para simplificar el modelo"""
    if label in ["Walk forward", "Walk backward"]:
        return "Walking"
    elif label in ["Get up", "Sit down"]:
        return "Transition"
    else:
        return label

def process_video(video_path, video_id, label_data, source="Entrega1"):
    """Extrae landmarks de cada frame y los une con etiquetas temporales."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: No se encontro: {video_path}")
        return pd.DataFrame()
    
    cap = cv2.VideoCapture(str(video_path))
    data = []

    # Obtener informacion del video
    total_frames_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Obtener el frame maximo de Label Studio
    max_frame_labelstudio = get_max_label_frame(video_id, label_data)
    
    # Calcular factor de conversion
    frame_ratio = (max_frame_labelstudio / total_frames_opencv) if (max_frame_labelstudio and total_frames_opencv) else 1.0
    
    print(f"  Frames OpenCV: {total_frames_opencv}, Label Studio max: {max_frame_labelstudio}, FPS: {fps:.2f}")
    print(f"  Ratio: {frame_ratio:.4f}, Source: {source}")
    
    frame_idx_opencv = 0
    pbar = tqdm(total=total_frames_opencv, desc=f"  {video_path.name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Convertir el frame de OpenCV al indice de Label Studio
            frame_idx_labelstudio = int(frame_idx_opencv * frame_ratio)
            
            row = {
                'video_id': video_id, 
                'source': source,  # Nueva columna para identificar el origen
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

            # Calidad: media de visibility y numero de landmarks visibles
            row['mean_visibility'] = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0
            row['num_visible_lms'] = int(sum(1 for v in vis_vals if v >= 0.5))

            # Centro de caderas y escala de torso
            try:
                lh, rh = landmarks[23], landmarks[24]
                ls, rs = landmarks[11], landmarks[12]
                hip_cx = (lh.x + rh.x) / 2.0
                hip_cy = (lh.y + rh.y) / 2.0
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

            # Extraer etiqueta y combinar clases
            original_label = extract_label_for_frame(video_id, frame_idx_labelstudio, label_data)
            row['label'] = combine_classes(original_label)
            data.append(row)

        frame_idx_opencv += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return pd.DataFrame(data)

# === FUNCION PARA ENRIQUECER DATOS ===

def enrich_dataset(df):
    """Agrega features derivados al dataset"""
    print("\nEnriqueciendo dataset con features derivados...")
    
    # Asegurar fps valido por video
    df["fps_eff"] = df.groupby(["source", "video_id"])["fps"].transform(
        lambda s: s.fillna(s.median()).replace(0, s.median()).fillna(30)
    )

    # Velocidades para landmarks clave
    keys = [15, 16, 25, 26, 27, 28]  # munecas, rodillas, tobillos
    print(f"  Calculando velocidades para landmarks: {keys}")
    for i in keys:
        dx = df.groupby(["source", "video_id"])[f"x_{i}"].diff()
        dy = df.groupby(["source", "video_id"])[f"y_{i}"].diff()
        df[f"speed_{i}"] = np.sqrt(dx.fillna(0)**2 + dy.fillna(0)**2) * df["fps_eff"]

    # Funcion para calcular angulo
    def angle_deg(ax, ay, bx, by, cx, cy):
        BAx, BAy = ax - bx, ay - by
        BCx, BCy = cx - bx, cy - by
        num = BAx * BCx + BAy * BCy
        den = np.sqrt(BAx**2 + BAy**2) * np.sqrt(BCx**2 + BCy**2)
        cosang = np.clip(num / np.where(den == 0, np.nan, den), -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    # Angulos de articulaciones
    print("  Calculando angulos de articulaciones...")
    
    df["knee_left_deg"] = angle_deg(
        df["x_23"], df["y_23"], df["x_25"], df["y_25"], df["x_27"], df["y_27"]
    )
    df["knee_right_deg"] = angle_deg(
        df["x_24"], df["y_24"], df["x_26"], df["y_26"], df["x_28"], df["y_28"]
    )
    df["elbow_left_deg"] = angle_deg(
        df["x_11"], df["y_11"], df["x_13"], df["y_13"], df["x_15"], df["y_15"]
    )
    df["elbow_right_deg"] = angle_deg(
        df["x_12"], df["y_12"], df["x_14"], df["y_14"], df["x_16"], df["y_16"]
    )

    # Segmentos contiguos por etiqueta
    print("  Creando segmentos por etiqueta...")
    df["segment_id"] = (
        df["label"].ne(df.groupby(["source", "video_id"])["label"].shift())
        .groupby([df["source"], df["video_id"]]).cumsum()
    )

    # Marca de baja calidad
    if "mean_visibility" in df and "num_visible_lms" in df:
        df["low_quality"] = (df["mean_visibility"] < 0.5) | (df["num_visible_lms"] < 15)
        n_low = df["low_quality"].sum()
        print(f"  Frames de baja calidad: {n_low} ({100*n_low/len(df):.1f}%)")

    return df

# === PROCESAMIENTO PRINCIPAL ===

def main():
    print("="*70)
    print("PROCESAMIENTO DE TODOS LOS VIDEOS DEL PROYECTO")
    print("="*70)
    
    all_data = []
    
    # === 1. PROCESAR ENTREGA 1 ===
    print("\n1. PROCESANDO ENTREGA 1 (Videos APO)")
    print("-"*70)
    
    label_data_e1 = load_label_data(ENTREGA1_LABEL_FILE)
    
    if label_data_e1 and ENTREGA1_VIDEOS_DIR.exists():
        # Crear mapeo de videos
        video_mapping_e1 = {}
        for entry in label_data_e1:
            json_video_name = entry["file_upload"].split("/")[-1]
            base_name = json_video_name.split("-")[-1]
            video_num = base_name.replace("Video_", "").replace(".mp4", "")
            real_video_name = f"Video {video_num}.mp4"
            video_mapping_e1[entry["id"]] = real_video_name
        
        # Procesar cada video
        for entry in label_data_e1:
            video_id = entry["id"]
            expected_video_name = video_mapping_e1[video_id]
            video_path = ENTREGA1_VIDEOS_DIR / expected_video_name
            
            if video_path.exists():
                print(f"\nProcesando {expected_video_name} (ID {video_id})...")
                df = process_video(video_path, video_id, label_data_e1, source="Entrega1")
                if not df.empty:
                    all_data.append(df)
            else:
                print(f"Advertencia: No se encontro: {expected_video_name}")
    else:
        print("Advertencia: No se encontraron datos de Entrega 1")
    
    # === 2. PROCESAR ENTREGA 2 - FAMILIA DIEGO ===
    print("\n2. PROCESANDO ENTREGA 2 - VIDEOS FAMILIA DIEGO")
    print("-"*70)
    
    label_data_e2_familia = load_label_data(ENTREGA2_FAMILIA_LABEL_FILE)
    
    if label_data_e2_familia and ENTREGA2_FAMILIA_VIDEOS_DIR.exists():
        for entry in label_data_e2_familia:
            video_id = entry["id"]
            # El nombre del archivo viene en "file_upload"
            video_filename = entry["file_upload"].split("/")[-1]
            # Extraer el nombre real del video (ej: "7a2d1d80-vid1.mp4" -> "vid1.mp4")
            video_name = video_filename.split("-", 1)[-1]
            video_path = ENTREGA2_FAMILIA_VIDEOS_DIR / video_name
            
            if video_path.exists():
                print(f"\nProcesando {video_name} (ID {video_id})...")
                df = process_video(video_path, video_id, label_data_e2_familia, source="Entrega2_FamiliaDiego")
                if not df.empty:
                    all_data.append(df)
            else:
                print(f"Advertencia: No se encontro: {video_name}")
    else:
        print("Advertencia: No se encontraron datos de Entrega 2 - Familia Diego")
    
    # === 3. PROCESAR ENTREGA 2 - OTRO GRUPO ===
    print("\n3. PROCESANDO ENTREGA 2 - VIDEOS OTRO GRUPO")
    print("-"*70)
    
    label_data_e2_otro = load_label_data(ENTREGA2_OTRO_LABEL_FILE)
    
    if label_data_e2_otro and ENTREGA2_OTRO_VIDEOS_DIR.exists():
        for entry in label_data_e2_otro:
            video_id = entry["id"]
            video_filename = entry["file_upload"].split("/")[-1]
            video_name = video_filename.split("-", 1)[-1]
            video_path = ENTREGA2_OTRO_VIDEOS_DIR / video_name
            
            if video_path.exists():
                print(f"\nProcesando {video_name} (ID {video_id})...")
                df = process_video(video_path, video_id, label_data_e2_otro, source="Entrega2_OtroGrupo")
                if not df.empty:
                    all_data.append(df)
            else:
                print(f"Advertencia: No se encontro: {video_name}")
    else:
        print("Advertencia: No se encontraron datos de Entrega 2 - Otro Grupo")
    
    # === 4. COMBINAR Y GUARDAR ===
    print("\n" + "="*70)
    print("COMBINANDO Y GUARDANDO DATOS")
    print("="*70)
    
    if all_data:
        # Combinar todos los DataFrames
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Guardar dataset sin enriquecer
        final_df.to_csv(str(OUTPUT_CSV), index=False)
        print(f"\nDataset base guardado: {OUTPUT_CSV.name}")
        print(f"   Total de frames: {len(final_df):,}")
        print(f"   Total de videos: {final_df['video_id'].nunique()}")
        print(f"   Fuentes: {final_df['source'].unique().tolist()}")
        
        # Distribucion por fuente
        print("\n   Distribucion por fuente:")
        for source in final_df['source'].unique():
            count = len(final_df[final_df['source'] == source])
            videos = final_df[final_df['source'] == source]['video_id'].nunique()
            print(f"      {source:25s}: {count:6,} frames ({videos} videos)")
        
        # Distribucion de etiquetas
        print("\n   Distribucion de etiquetas:")
        label_counts = final_df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"      {label:25s}: {count:6,} frames ({percentage:5.2f}%)")
        
        # Enriquecer dataset
        enriched_df = enrich_dataset(final_df)
        
        # Guardar dataset enriquecido
        enriched_df.to_csv(str(OUTPUT_ENRICHED_CSV), index=False)
        print(f"\nDataset enriquecido guardado: {OUTPUT_ENRICHED_CSV.name}")
        print(f"   Shape: {enriched_df.shape}")
        print(f"   Columnas agregadas: velocidades, angulos, segmentos, calidad")
        
    else:
        print("\nAdvertencia: No se genero ningun dataset (no se encontraron videos)")
    
    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()
