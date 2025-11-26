"""
Script para evaluar métricas del modelo sobre videos con etiquetas reales.
Calcula F1-Score, Accuracy, Precision, Recall y genera reportes detallados.
VERSIÓN CON CLASES COMBINADAS: Walking (Walk forward + Walk backward), Transition (Get up + Sit down)
"""

import json
import math
import os
import warnings
from collections import Counter
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

# === CONFIGURACIÓN ===
SCRIPT_DIR = Path(__file__).parent.absolute()

MODEL_PATH = SCRIPT_DIR / "models" / "best_model.joblib"
SCALER_PATH = SCRIPT_DIR / "models" / "scaler.joblib"
LABEL_ENCODER_PATH = SCRIPT_DIR / "models" / "label_encoder.joblib"

# Configurar fuentes de videos
VIDEO_SOURCES = [
    {
        'name': 'Entrega1',
        'videos_dir': SCRIPT_DIR / ".." / "Entrega 1" / "Videos APO" / "Videos APO",
        'label_file': SCRIPT_DIR / ".." / "Entrega 1" / "project-label-studio.json"
    },
    {
        'name': 'Entrega2_FamiliaDiego',
        'videos_dir': SCRIPT_DIR / "videos familiaDiego",
        'label_file': SCRIPT_DIR / "videos familiaDiego" / "project-4-at-2025-11-02-12-56-089d6ad2.json"
    },
    {
        'name': 'Entrega2_OtroGrupo',
        'videos_dir': SCRIPT_DIR / "videos otroGrupo",
        'label_file': SCRIPT_DIR / "videos otroGrupo" / "project-5-at-2025-11-02-14-50-27ad6e06.json"
    }
]

OUTPUT_REPORT = SCRIPT_DIR / "video_evaluation_report.txt"
OUTPUT_CSV = SCRIPT_DIR / "video_evaluation_details.csv"

# Parámetros
SAMPLE_EVERY_N_FRAMES = 5  # Evaluar cada N frames para agilizar
CONFIDENCE_THRESHOLD = 0.0  # Sin umbral (todas las predicciones cuentan)

# === CARGAR MODELO ===
print("🔄 Cargando modelo y transformadores...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Modelo cargado exitosamente")
    print(f"   Clases: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

# === MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === FUNCIONES DE EXTRACCIÓN ===
def angle_deg(x1, y1, x2, y2, x3, y3):
    """Calcula ángulo en grados entre tres puntos"""
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    return math.degrees(math.acos(cos_angle))

def extract_features_from_landmarks(landmarks, prev_landmarks, fps):
    """Extrae las 16 features biomecánicas"""
    features = {}
    
    xs, ys, vis_vals = [], [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
        vis_vals.append(lm.visibility)
    
    if not xs or not ys:
        return None
    
    # hip_center_x, hip_center_y, torso_scale
    try:
        lh, rh = landmarks[23], landmarks[24]
        ls, rs = landmarks[11], landmarks[12]
        hip_cx = (lh.x + rh.x) / 2.0
        hip_cy = (lh.y + rh.y) / 2.0
        d_l = math.hypot(ls.x - lh.x, ls.y - lh.y)
        d_r = math.hypot(rs.x - rh.x, rs.y - rh.y)
        torso_scale = max((d_l + d_r) / 2.0, 1e-6)
        features['hip_center_x'] = hip_cx
        features['hip_center_y'] = hip_cy
        features['torso_scale'] = torso_scale
    except Exception:
        hip_cx, hip_cy, torso_scale = None, None, None
        features['hip_center_x'] = None
        features['hip_center_y'] = None
        features['torso_scale'] = None
    
    # bbox_area, bbox_aspect
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    features['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
    features['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else None
    
    # knee_left_deg, knee_right_deg
    features['knee_left_deg'] = angle_deg(
        landmarks[23].x, landmarks[23].y,
        landmarks[25].x, landmarks[25].y,
        landmarks[27].x, landmarks[27].y
    )
    features['knee_right_deg'] = angle_deg(
        landmarks[24].x, landmarks[24].y,
        landmarks[26].x, landmarks[26].y,
        landmarks[28].x, landmarks[28].y
    )
    
    # elbow_left_deg, elbow_right_deg
    features['elbow_left_deg'] = angle_deg(
        landmarks[11].x, landmarks[11].y,
        landmarks[13].x, landmarks[13].y,
        landmarks[15].x, landmarks[15].y
    )
    features['elbow_right_deg'] = angle_deg(
        landmarks[12].x, landmarks[12].y,
        landmarks[14].x, landmarks[14].y,
        landmarks[16].x, landmarks[16].y
    )
    
    # speed_15, speed_16 (velocidad muñecas)
    if prev_landmarks is not None and fps > 0:
        dx_15 = landmarks[15].x - prev_landmarks[15].x
        dy_15 = landmarks[15].y - prev_landmarks[15].y
        features['speed_15'] = math.sqrt(dx_15**2 + dy_15**2) * fps
        
        dx_16 = landmarks[16].x - prev_landmarks[16].x
        dy_16 = landmarks[16].y - prev_landmarks[16].y
        features['speed_16'] = math.sqrt(dx_16**2 + dy_16**2) * fps
    else:
        features['speed_15'] = 0.0
        features['speed_16'] = 0.0
    
    # velocidad_centro_cuerpo
    if prev_landmarks is not None and hip_cx is not None and fps > 0:
        prev_hip_cx = (prev_landmarks[23].x + prev_landmarks[24].x) / 2.0
        prev_hip_cy = (prev_landmarks[23].y + prev_landmarks[24].y) / 2.0
        dx_center = hip_cx - prev_hip_cx
        dy_center = hip_cy - prev_hip_cy
        features['velocidad_centro_cuerpo'] = math.sqrt(dx_center**2 + dy_center**2) * fps
    else:
        features['velocidad_centro_cuerpo'] = 0.0
    
    # apertura_piernas_norm
    if torso_scale and torso_scale > 0:
        apertura_piernas = abs(landmarks[27].x - landmarks[28].x)
        features['apertura_piernas_norm'] = apertura_piernas / torso_scale
    else:
        features['apertura_piernas_norm'] = None
    
    # ancho_hombros
    features['ancho_hombros'] = abs(landmarks[11].x - landmarks[12].x)
    
    # altura_normalizada
    try:
        nose = landmarks[0]
        ankle_l, ankle_r = landmarks[27], landmarks[28]
        ankle_avg_y = (ankle_l.y + ankle_r.y) / 2.0
        features['altura_normalizada'] = abs(nose.y - ankle_avg_y)
    except Exception:
        features['altura_normalizada'] = None
    
    # dist_vertical_cabeza_cadera
    try:
        nose = landmarks[0]
        hip_avg_y = (landmarks[23].y + landmarks[24].y) / 2.0
        features['dist_vertical_cabeza_cadera'] = abs(nose.y - hip_avg_y)
    except Exception:
        features['dist_vertical_cabeza_cadera'] = None
    
    return features

def predict_activity(features_dict, model, scaler, label_encoder):
    """Realiza predicción"""
    df = pd.DataFrame([features_dict])
    expected_features = scaler.feature_names_in_
    df_filtered = df[expected_features].fillna(0)
    
    try:
        X_scaled = scaler.transform(df_filtered)
        X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)
        
        y_pred = model.predict(X_scaled_df)
        y_proba = model.predict_proba(X_scaled_df)
        
        label = label_encoder.inverse_transform(y_pred)[0]
        confidence = np.max(y_proba)
        
        return label, confidence
    except Exception as e:
        return None, 0.0

# === FUNCIONES DE LABEL STUDIO ===
def load_label_data(label_file):
    """Carga el archivo JSON de Label Studio"""
    label_file = Path(label_file)
    if not label_file.exists():
        return []
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_max_label_frame(video_id, label_data):
    """Obtiene el frame máximo etiquetado"""
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
    """Obtiene la etiqueta real de un frame"""
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

# === EVALUAR VIDEO ===
def evaluate_video(video_path, video_id, label_data, source_name):
    """Evalúa un video completo y retorna métricas"""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"⚠️ No se encontró: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Obtener ratio de conversión de frames
    max_frame_labelstudio = get_max_label_frame(video_id, label_data)
    frame_ratio = (max_frame_labelstudio / total_frames) if (max_frame_labelstudio and total_frames) else 1.0
    
    y_true = []
    y_pred = []
    confidences = []
    
    frame_idx_opencv = 0
    prev_landmarks = None
    
    pbar = tqdm(total=total_frames, desc=f"  {video_path.name}", leave=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Samplear cada N frames para agilizar
        if frame_idx_opencv % SAMPLE_EVERY_N_FRAMES == 0:
            frame_idx_labelstudio = int(frame_idx_opencv * frame_ratio)
            original_label = extract_label_for_frame(video_id, frame_idx_labelstudio, label_data)
            true_label = combine_classes(original_label)  # COMBINAR CLASES
            
            # Ignorar frames sin etiqueta
            if true_label != "Unlabeled":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    features = extract_features_from_landmarks(landmarks, prev_landmarks, fps)
                    
                    if features:
                        pred_label, confidence = predict_activity(features, model, scaler, label_encoder)
                        
                        if pred_label:
                            # IMPORTANTE: Combinar también las predicciones
                            pred_label_combined = combine_classes(pred_label)
                            y_true.append(true_label)
                            y_pred.append(pred_label_combined)
                            confidences.append(confidence)
                    
                    prev_landmarks = landmarks
        
        frame_idx_opencv += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    if len(y_true) == 0:
        print(f"⚠️ No se obtuvieron predicciones válidas para {video_path.name}")
        return None
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'source': source_name,
        'video_name': video_path.name,
        'video_id': video_id,
        'total_samples': len(y_true),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confidence_mean': np.mean(confidences),
        'confidence_std': np.std(confidences),
        'y_true': y_true,
        'y_pred': y_pred
    }

# === PROCESAMIENTO PRINCIPAL ===
def main():
    print("="*90)
    print("📊 EVALUACIÓN DE VIDEOS CON MÉTRICAS (CLASES COMBINADAS)")
    print("="*90)
    print("   Walking = Walk forward + Walk backward")
    print("   Transition = Get up + Sit down")
    print("="*90)
    
    all_results = []
    all_y_true = []
    all_y_pred = []
    
    for source_config in VIDEO_SOURCES:
        source_name = source_config['name']
        videos_dir = Path(source_config['videos_dir'])
        label_file = Path(source_config['label_file'])
        
        print(f"\n🎬 Procesando fuente: {source_name}")
        print(f"   Videos dir: {videos_dir}")
        print(f"   Labels: {label_file}")
        
        if not videos_dir.exists():
            print(f"   ⚠️ Directorio no encontrado, saltando...")
            continue
        
        if not label_file.exists():
            print(f"   ⚠️ Archivo de etiquetas no encontrado, saltando...")
            continue
        
        label_data = load_label_data(label_file)
        if not label_data:
            print(f"   ⚠️ No se pudieron cargar las etiquetas, saltando...")
            continue
        
        # Crear mapeo de videos
        video_mapping = {}
        for entry in label_data:
            video_filename = entry["file_upload"].split("/")[-1]
            
            # Manejar diferentes formatos de nombres
            if source_name == 'Entrega1':
                # Formato: "Video-1.mp4" -> "Video 1.mp4"
                base_name = video_filename.split("-")[-1]
                video_num = base_name.replace("Video_", "").replace(".mp4", "")
                real_video_name = f"Video {video_num}.mp4"
            else:
                # Formato: "7a2d1d80-vid1.mp4" -> "vid1.mp4"
                real_video_name = video_filename.split("-", 1)[-1]
            
            video_mapping[entry["id"]] = real_video_name
        
        print(f"   Videos a evaluar: {len(video_mapping)}")
        
        # Evaluar cada video
        for video_id, video_name in video_mapping.items():
            video_path = videos_dir / video_name
            
            if not video_path.exists():
                print(f"   ⚠️ No encontrado: {video_name}")
                continue
            
            print(f"\n   📹 Evaluando: {video_name}")
            result = evaluate_video(video_path, video_id, label_data, source_name)
            
            if result:
                all_results.append(result)
                all_y_true.extend(result['y_true'])
                all_y_pred.extend(result['y_pred'])
                
                print(f"      ✅ Samples: {result['total_samples']:4d} | "
                      f"Acc: {result['accuracy']:.3f} | "
                      f"F1: {result['f1_score']:.3f} | "
                      f"Prec: {result['precision']:.3f} | "
                      f"Rec: {result['recall']:.3f}")
    
    # === GENERAR REPORTE ===
    if not all_results:
        print("\n❌ No se obtuvieron resultados")
        return
    
    print("\n" + "="*90)
    print("📈 REPORTE DE MÉTRICAS GLOBAL")
    print("="*90)
    
    # Guardar en CSV
    df_results = pd.DataFrame([
        {
            'source': r['source'],
            'video_name': r['video_name'],
            'samples': r['total_samples'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1_score': r['f1_score'],
            'confidence_mean': r['confidence_mean'],
            'confidence_std': r['confidence_std']
        }
        for r in all_results
    ])
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Detalles guardados en: {OUTPUT_CSV}")
    
    # Reporte por fuente
    report_lines = []
    report_lines.append("="*90)
    report_lines.append("📊 MÉTRICAS POR FUENTE DE DATOS (CLASES COMBINADAS)")
    report_lines.append("="*90)
    report_lines.append("   Walking = Walk forward + Walk backward")
    report_lines.append("   Transition = Get up + Sit down")
    report_lines.append("="*90)
    
    for source_name in df_results['source'].unique():
        source_data = df_results[df_results['source'] == source_name]
        
        report_lines.append(f"\n🎬 {source_name}")
        report_lines.append(f"   Videos evaluados: {len(source_data)}")
        report_lines.append(f"   Total samples: {source_data['samples'].sum():,}")
        report_lines.append(f"   Accuracy promedio:  {source_data['accuracy'].mean():.4f} ± {source_data['accuracy'].std():.4f}")
        report_lines.append(f"   Precision promedio: {source_data['precision'].mean():.4f} ± {source_data['precision'].std():.4f}")
        report_lines.append(f"   Recall promedio:    {source_data['recall'].mean():.4f} ± {source_data['recall'].std():.4f}")
        report_lines.append(f"   F1-Score promedio:  {source_data['f1_score'].mean():.4f} ± {source_data['f1_score'].std():.4f}")
    
    # Métricas globales
    report_lines.append("\n" + "="*90)
    report_lines.append("🌍 MÉTRICAS GLOBALES (TODOS LOS VIDEOS)")
    report_lines.append("="*90)
    
    global_accuracy = accuracy_score(all_y_true, all_y_pred)
    global_precision = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    global_recall = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    global_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    
    report_lines.append(f"\n📊 Total videos evaluados: {len(all_results)}")
    report_lines.append(f"📊 Total samples: {len(all_y_true):,}")
    report_lines.append(f"\n🎯 Accuracy:  {global_accuracy:.4f}")
    report_lines.append(f"🎯 Precision: {global_precision:.4f}")
    report_lines.append(f"🎯 Recall:    {global_recall:.4f}")
    report_lines.append(f"🎯 F1-Score:  {global_f1:.4f}")
    
    # Reporte por clase
    report_lines.append("\n" + "="*90)
    report_lines.append("📋 REPORTE DETALLADO POR CLASE")
    report_lines.append("="*90)
    
    class_report = classification_report(
        all_y_true, all_y_pred,
        target_names=sorted(set(all_y_true)),
        digits=4,
        zero_division=0
    )
    report_lines.append(f"\n{class_report}")
    
    # Matriz de confusión
    report_lines.append("\n" + "="*90)
    report_lines.append("🔢 MATRIZ DE CONFUSIÓN")
    report_lines.append("="*90)
    
    classes = sorted(set(all_y_true))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    
    # Header
    header = "True \\ Pred |" + "|".join([f"{cls:12s}" for cls in classes])
    report_lines.append(f"\n{header}")
    report_lines.append("-" * len(header))
    
    # Filas
    for i, true_cls in enumerate(classes):
        row = f"{true_cls:12s}|" + "|".join([f"{cm[i][j]:12d}" for j in range(len(classes))])
        report_lines.append(row)
    
    # Top errores
    report_lines.append("\n" + "="*90)
    report_lines.append("❌ TOP 10 CONFUSIONES MÁS FRECUENTES")
    report_lines.append("="*90)
    
    errors = []
    for i in range(len(all_y_true)):
        if all_y_true[i] != all_y_pred[i]:
            errors.append((all_y_true[i], all_y_pred[i]))
    
    error_counts = Counter(errors)
    report_lines.append(f"\nTotal errores: {len(errors)} de {len(all_y_true)} ({100*len(errors)/len(all_y_true):.1f}%)")
    report_lines.append("")
    
    for (true_label, pred_label), count in error_counts.most_common(10):
        pct = 100 * count / len(errors)
        report_lines.append(f"   {true_label:20s} → {pred_label:20s}: {count:4d} veces ({pct:5.1f}%)")
    
    # Mejores y peores videos
    report_lines.append("\n" + "="*90)
    report_lines.append("🏆 TOP 5 VIDEOS CON MEJOR F1-SCORE")
    report_lines.append("="*90)
    
    df_sorted = df_results.sort_values('f1_score', ascending=False)
    for idx, row in df_sorted.head(5).iterrows():
        report_lines.append(f"\n   {row['video_name']:30s} ({row['source']:25s})")
        report_lines.append(f"      F1: {row['f1_score']:.4f} | Acc: {row['accuracy']:.4f} | Samples: {row['samples']}")
    
    report_lines.append("\n" + "="*90)
    report_lines.append("⚠️ TOP 5 VIDEOS CON MENOR F1-SCORE")
    report_lines.append("="*90)
    
    for idx, row in df_sorted.tail(5).iterrows():
        report_lines.append(f"\n   {row['video_name']:30s} ({row['source']:25s})")
        report_lines.append(f"      F1: {row['f1_score']:.4f} | Acc: {row['accuracy']:.4f} | Samples: {row['samples']}")
    
    # Estadísticas de confianza
    report_lines.append("\n" + "="*90)
    report_lines.append("📊 ESTADÍSTICAS DE CONFIANZA DEL MODELO")
    report_lines.append("="*90)
    
    all_confidences = []
    for r in all_results:
        all_confidences.append(r['confidence_mean'])
    
    report_lines.append(f"\n   Confianza promedio global: {np.mean(all_confidences):.4f}")
    report_lines.append(f"   Desviación estándar:       {np.std(all_confidences):.4f}")
    report_lines.append(f"   Confianza mínima:          {np.min(all_confidences):.4f}")
    report_lines.append(f"   Confianza máxima:          {np.max(all_confidences):.4f}")
    
    report_lines.append("\n" + "="*90)
    report_lines.append("✅ EVALUACIÓN COMPLETADA")
    report_lines.append("="*90)
    
    # Imprimir y guardar reporte
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n💾 Reporte completo guardado en: {OUTPUT_REPORT}")
    print(f"💾 Detalles CSV en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
    pose.close()
