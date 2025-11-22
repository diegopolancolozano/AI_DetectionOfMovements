"""
Script avanzado para procesar videos y generar video con predicciones
Crea un video de salida donde se ven las predicciones en tiempo real
"""

import math
import os
import warnings
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

# === CONFIGURACIÓN ===
MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"

VIDEO_PATH = "/home/oscar/Documents/APO III/AI_DetectionOfMovements/Entrega 1/Videos APO/mio.mp4"
OUTPUT_VIDEO_PATH = "video_2_analyzed.mp4"

# === PARÁMETROS ===
PREDICTION_BUFFER_SIZE = 5
CONFIDENCE_THRESHOLD = 0.65
STABILITY_FRAMES = 3

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
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === FUNCIONES ===
def angle_deg(x1, y1, x2, y2, x3, y3):
    """Calcula ángulo en grados entre tres puntos"""
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    return math.degrees(math.acos(cos_angle))

def extract_features_from_landmarks(landmarks, prev_landmarks, fps):
    """
    Extrae SOLO las 16 features biomecánicas usadas en el entrenamiento.
    Replica exactamente la lógica de extract_mediapipe_selected_features.py
    """
    features = {}
    
    # === EXTRAER COORDENADAS Y VISIBILIDAD ===
    xs, ys, vis_vals = [], [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
        vis_vals.append(lm.visibility)
    
    if not xs or not ys:
        return None
    
    # === 3-5. hip_center_x, hip_center_y, torso_scale ===
    try:
        lh, rh = landmarks[23], landmarks[24]   # left/right hip
        ls, rs = landmarks[11], landmarks[12]   # left/right shoulder
        hip_cx = (lh.x + rh.x) / 2.0
        hip_cy = (lh.y + rh.y) / 2.0
        d_l = math.hypot(ls.x - lh.x, ls.y - lh.y)
        d_r = math.hypot(rs.x - rh.x, rs.y - rh.y)
        torso_scale = max((d_l + d_r) / 2.0, 1e-6)
        features['hip_center_x'] = hip_cx
        features['hip_center_y'] = hip_cy
        features['torso_scale'] = torso_scale
    except Exception:
        hip_cx, hip_cy = None, None
        torso_scale = None
        features['hip_center_x'] = None
        features['hip_center_y'] = None
        features['torso_scale'] = None
    
    # === 6-7. bbox_area, bbox_aspect ===
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    features['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
    features['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else None
    
    # === 8-9. knee_left_deg, knee_right_deg ===
    features['knee_left_deg'] = angle_deg(
        landmarks[23].x, landmarks[23].y,  # cadera izq
        landmarks[25].x, landmarks[25].y,  # rodilla izq
        landmarks[27].x, landmarks[27].y   # tobillo izq
    )
    features['knee_right_deg'] = angle_deg(
        landmarks[24].x, landmarks[24].y,  # cadera der
        landmarks[26].x, landmarks[26].y,  # rodilla der
        landmarks[28].x, landmarks[28].y   # tobillo der
    )
    
    # === 10-11. elbow_left_deg, elbow_right_deg ===
    features['elbow_left_deg'] = angle_deg(
        landmarks[11].x, landmarks[11].y,  # hombro izq
        landmarks[13].x, landmarks[13].y,  # codo izq
        landmarks[15].x, landmarks[15].y   # muñeca izq
    )
    features['elbow_right_deg'] = angle_deg(
        landmarks[12].x, landmarks[12].y,  # hombro der
        landmarks[14].x, landmarks[14].y,  # codo der
        landmarks[16].x, landmarks[16].y   # muñeca der
    )
    
    # === 12-13. speed_15, speed_16 (velocidad muñecas) ===
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
    
    # === 14. velocidad_centro_cuerpo ===
    if prev_landmarks is not None and hip_cx is not None and fps > 0:
        prev_hip_cx = (prev_landmarks[23].x + prev_landmarks[24].x) / 2.0
        prev_hip_cy = (prev_landmarks[23].y + prev_landmarks[24].y) / 2.0
        dx_center = hip_cx - prev_hip_cx
        dy_center = hip_cy - prev_hip_cy
        features['velocidad_centro_cuerpo'] = math.sqrt(dx_center**2 + dy_center**2) * fps
    else:
        features['velocidad_centro_cuerpo'] = 0.0
    
    # === 15. apertura_piernas_norm ===
    if torso_scale and torso_scale > 0:
        apertura_piernas = abs(landmarks[27].x - landmarks[28].x)  # tobillos
        features['apertura_piernas_norm'] = apertura_piernas / torso_scale
    else:
        features['apertura_piernas_norm'] = None
    
    # === 16. ancho_hombros ===
    features['ancho_hombros'] = abs(landmarks[11].x - landmarks[12].x)
    
    # === 17. altura_normalizada ===
    try:
        nose = landmarks[0]
        ankle_l, ankle_r = landmarks[27], landmarks[28]
        ankle_avg_y = (ankle_l.y + ankle_r.y) / 2.0
        features['altura_normalizada'] = abs(nose.y - ankle_avg_y)
    except Exception:
        features['altura_normalizada'] = None
    
    # === 18. dist_vertical_cabeza_cadera ===
    try:
        nose = landmarks[0]
        hip_avg_y = (landmarks[23].y + landmarks[24].y) / 2.0
        features['dist_vertical_cabeza_cadera'] = abs(nose.y - hip_avg_y)
    except Exception:
        features['dist_vertical_cabeza_cadera'] = None
    
    return features

def postural_check(label, features):
    """Valida si las features concuerdan con la clase predicha"""
    if features.get('knee_left_deg') is None or features.get('knee_right_deg') is None:
        return True, 1.0
    
    knee_avg = (features['knee_left_deg'] + features['knee_right_deg']) / 2
    
    if label == 'Sitting' and knee_avg > 100:
        return False, 0.4
    
    if label == 'Standing' and knee_avg < 70:
        return False, 0.4
    
    return True, 1.0

def predict_activity(features_dict, model, scaler, label_encoder):
    """Realiza predicción con validación postural"""
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
        
        is_valid, postural_factor = postural_check(label, features_dict)
        if not is_valid:
            confidence *= postural_factor
        
        return label, confidence
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return "Error", 0.0

# === PROCESAR VIDEO ===
print(f"\n📹 Abriendo video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ No se pudo abrir el video")
    exit(1)

# Propiedades del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Video abierto")
print(f"   Resolución: {width}x{height}")
print(f"   FPS: {fps:.1f}")
print(f"   Total frames: {total_frames}")

# Configurar VideoWriter para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ No se pudo crear el VideoWriter")
    cap.release()
    exit(1)

frame_count = 0
prev_landmarks = None
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
current_label = None
stability_counter = 0
predictions_log = []

print(f"\n🎬 Procesando video...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video terminado")
            break
        
        frame_count += 1
        
        # Mostrar progreso
        if frame_count % 50 == 0:
            pct = 100 * frame_count / total_frames
            print(f"   Progreso: {frame_count}/{total_frames} ({pct:.0f}%)")
        
        # Procesar con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        frame_display = frame.copy()
        
        if results.pose_landmarks:
            # Dibujar skeleton
            mp_drawing.draw_landmarks(
                frame_display,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            features = extract_features_from_landmarks(landmarks, prev_landmarks, fps)
            
            if features:
                label, confidence = predict_activity(features, model, scaler, label_encoder)
                
                # Agregar al buffer
                prediction_buffer.append((label, confidence))
                
                # Lógica de estabilidad
                if confidence >= CONFIDENCE_THRESHOLD:
                    if label == current_label:
                        stability_counter += 1
                    else:
                        if stability_counter >= STABILITY_FRAMES:
                            current_label = label
                            stability_counter = 1
                        else:
                            stability_counter = 1
                else:
                    if len(prediction_buffer) >= PREDICTION_BUFFER_SIZE:
                        labels = [p[0] for p in prediction_buffer]
                        label_counts = Counter(labels)
                        most_common = label_counts.most_common(1)[0][0]
                        
                        if most_common == current_label:
                            stability_counter += 1
                        else:
                            if stability_counter >= STABILITY_FRAMES:
                                current_label = most_common
                                stability_counter = 1
                            else:
                                stability_counter = 1
                    else:
                        stability_counter = 0
                
                # Registrar predicción
                predictions_log.append({
                    'frame': frame_count,
                    'label': label,
                    'confidence': confidence,
                    'current': current_label,
                    'stability': stability_counter
                })
                
                # === DIBUJAR EN FRAME ===
                h, w = frame_display.shape[:2]
                
                # Panel de fondo
                cv2.rectangle(frame_display, (0, 0), (750, 200), (0, 0, 0), -1)
                cv2.rectangle(frame_display, (0, 0), (750, 200), (0, 255, 0), 2)
                
                # Frame número
                cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", (15, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Actividad (GRANDE Y DESTACADA)
                color = (0, 255, 0) if stability_counter >= STABILITY_FRAMES else (0, 165, 255)
                cv2.putText(frame_display, f"ACTIVIDAD: {current_label or '?'}", (15, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Predicción actual
                cv2.putText(frame_display, f"Pred: {label} ({confidence:.0%})", (15, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                
                # Estabilidad
                stab_color = (0, 255, 0) if stability_counter >= STABILITY_FRAMES else (0, 165, 255)
                cv2.putText(frame_display, f"Estabilidad: {stability_counter}/{STABILITY_FRAMES}", (15, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 1)
                
                # Buffer votos
                if len(prediction_buffer) > 0:
                    buffer_votes = Counter([p[0] for p in prediction_buffer])
                    votes_text = " | ".join([f"{k}: {v}" for k, v in buffer_votes.most_common(2)])
                    cv2.putText(frame_display, f"Votos: {votes_text}", (15, 190),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            prev_landmarks = landmarks
        
        # Escribir frame en video de salida
        out.write(frame_display)

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido")

finally:
    cap.release()
    out.release()
    
    # === GENERAR REPORTE ===
    print("\n" + "="*90)
    print("📊 REPORTE DE ANÁLISIS DEL VIDEO")
    print("="*90)
    
    if predictions_log:
        labels = [p['label'] for p in predictions_log]
        current_labels = [p['current'] for p in predictions_log if p['current']]
        
        label_counts = Counter(labels)
        current_counts = Counter(current_labels)
        
        print(f"\n✅ Frames procesados: {frame_count}/{total_frames}")
        print(f"✅ Total predicciones: {len(predictions_log)}")
        
        print(f"\n📈 Distribución de predicciones iniciales (modelo):")
        total_pred = len(predictions_log)
        for label, count in label_counts.most_common():
            pct = 100 * count / total_pred
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            print(f"   {label:20s} {count:3d} ({pct:5.1f}%) {bar}")
        
        print(f"\n📈 Distribución después de suavizado:")
        if current_labels:
            total_smooth = len(current_labels)
            for label, count in current_counts.most_common():
                pct = 100 * count / total_smooth
                bar_length = int(pct / 2)
                bar = "█" * bar_length
                print(f"   {label:20s} {count:3d} ({pct:5.1f}%) {bar}")
        else:
            print("   (No se estabilizaron predicciones)")
        
        # Analizar transiciones
        transitions = []
        for i in range(1, len(predictions_log)):
            if predictions_log[i]['current'] and predictions_log[i-1]['current']:
                if predictions_log[i]['current'] != predictions_log[i-1]['current']:
                    transitions.append({
                        'frame': predictions_log[i]['frame'],
                        'from': predictions_log[i-1]['current'],
                        'to': predictions_log[i]['current']
                    })
        
        print(f"\n🔄 Cambios de actividad detectados: {len(transitions)}")
        if transitions:
            for t in transitions[:10]:  # Mostrar primeras 10
                print(f"   Frame {t['frame']:4d}: {t['from']:20s} → {t['to']:20s}")
            if len(transitions) > 10:
                print(f"   ... y {len(transitions)-10} cambios más")
        
        # Estadísticas de confianza
        confidences = [p['confidence'] for p in predictions_log]
        print(f"\n📊 Estadísticas de confianza:")
        print(f"   Promedio: {np.mean(confidences):.1%}")
        print(f"   Mínima:   {np.min(confidences):.1%}")
        print(f"   Máxima:   {np.max(confidences):.1%}")
        print(f"   Mediana:  {np.median(confidences):.1%}")
        
        # Predicciones con baja confianza
        low_confidence = [p for p in predictions_log if p['confidence'] < 0.5]
        print(f"   Predicciones con baja confianza (<50%): {len(low_confidence)} ({100*len(low_confidence)/len(predictions_log):.1f}%)")
        
        # Guardar log detallado
        import json
        log_file = 'video_2_analysis_log.json'
        with open(log_file, 'w') as f:
            data = {
                'video': VIDEO_PATH,
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'duration': f"{total_frames/fps:.1f}s",
                'predictions_count': len(predictions_log),
                'transitions': len(transitions),
                'confidence_avg': float(np.mean(confidences)),
                'predictions': []
            }
            
            for p in predictions_log:
                p['confidence'] = float(p['confidence'])
                data['predictions'].append(p)
            
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Archivos generados:")
        print(f"   Video: {OUTPUT_VIDEO_PATH}")
        print(f"   Log JSON: {log_file}")
        
        # Mostrar tamaño del video
        if os.path.exists(OUTPUT_VIDEO_PATH):
            size_mb = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024*1024)
            print(f"   Tamaño video: {size_mb:.1f} MB")
    
    print("="*90)
    print("✅ Procesamiento completado")
