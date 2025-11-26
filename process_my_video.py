"""
Script para procesar un video y generar video con predicciones
Versión adaptada para Windows
"""

import math
import os
import sys
import warnings
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

# === CONFIGURACIÓN ===
MODEL_PATH = "Entrega 2 Y 3 (Dodigo actualizado)/models/best_model.joblib"
SCALER_PATH = "Entrega 2 Y 3 (Dodigo actualizado)/models/scaler.joblib"
LABEL_ENCODER_PATH = "Entrega 2 Y 3 (Dodigo actualizado)/models/label_encoder.joblib"

# Pedir ruta del video al usuario
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    print("\n" + "="*70)
    print("PROCESADOR DE VIDEO CON DETECCIÓN DE MOVIMIENTOS")
    print("="*70)
    print("\nIngresa la ruta completa del video a procesar")
    print("(o arrastra el archivo aquí y presiona Enter)")
    print("\nEjemplo: C:\\Users\\tu_usuario\\Videos\\mi_video.mp4")
    print("-"*70)
    VIDEO_PATH = input("Ruta del video: ").strip().strip('"')

if not os.path.exists(VIDEO_PATH):
    print(f"\n❌ Error: No se encontró el video en: {VIDEO_PATH}")
    sys.exit(1)

# Nombre del video de salida
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = f"{video_name}_analyzed.mp4"

# === PARÁMETROS OPTIMIZADOS ===
PREDICTION_BUFFER_SIZE = 7  # Aumentado para más suavizado
CONFIDENCE_THRESHOLD = 0.55  # Reducido para aceptar más predicciones
STABILITY_FRAMES = 1  # Solo 1 frame para confirmar (inmediato)
TEMPORAL_WINDOW = 5  # Ventana más amplia (5 frames antes/después)

# === CARGAR MODELO ===
print("\n🔄 Cargando modelo y transformadores...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Modelo cargado exitosamente")
    print(f"   Clases: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    sys.exit(1)

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
    """Extrae las 16 features biomecánicas usadas en el entrenamiento"""
    features = {}
    
    # === EXTRAER COORDENADAS Y VISIBILIDAD ===
    xs, ys, vis_vals = [], [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
        vis_vals.append(lm.visibility)
    
    if not xs or not ys:
        return None
    
    # === hip_center_x, hip_center_y, torso_scale ===
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
        hip_cx, hip_cy = None, None
        torso_scale = None
        features['hip_center_x'] = None
        features['hip_center_y'] = None
        features['torso_scale'] = None
    
    # === bbox_area, bbox_aspect ===
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    features['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
    features['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else None
    
    # === knee_left_deg, knee_right_deg ===
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
    
    # === elbow_left_deg, elbow_right_deg ===
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
    
    # === speed_15, speed_16 (velocidad muñecas) ===
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
    
    # === velocidad_centro_cuerpo ===
    if prev_landmarks is not None and hip_cx is not None and fps > 0:
        prev_hip_cx = (prev_landmarks[23].x + prev_landmarks[24].x) / 2.0
        prev_hip_cy = (prev_landmarks[23].y + prev_landmarks[24].y) / 2.0
        dx_center = hip_cx - prev_hip_cx
        dy_center = hip_cy - prev_hip_cy
        features['velocidad_centro_cuerpo'] = math.sqrt(dx_center**2 + dy_center**2) * fps
    else:
        features['velocidad_centro_cuerpo'] = 0.0
    
    # === apertura_piernas_norm ===
    if torso_scale and torso_scale > 0:
        apertura_piernas = abs(landmarks[27].x - landmarks[28].x)
        features['apertura_piernas_norm'] = apertura_piernas / torso_scale
    else:
        features['apertura_piernas_norm'] = None
    
    # === ancho_hombros ===
    features['ancho_hombros'] = abs(landmarks[11].x - landmarks[12].x)
    
    # === altura_normalizada ===
    try:
        nose = landmarks[0]
        ankle_l, ankle_r = landmarks[27], landmarks[28]
        ankle_avg_y = (ankle_l.y + ankle_r.y) / 2.0
        features['altura_normalizada'] = abs(nose.y - ankle_avg_y)
    except Exception:
        features['altura_normalizada'] = None
    
    # === dist_vertical_cabeza_cadera ===
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
        print(f"⚠️ Error en predicción: {e}")
        return "Error", 0.0

def temporal_smoothing(temporal_predictions, current_frame, window_size=3):
    """
    Suavizado temporal mejorado: considera predicciones de frames cercanos
    con ponderación por distancia y confianza
    """
    if len(temporal_predictions) < window_size:
        return None, 0.0
    
    # Obtener índices de la ventana temporal
    start_idx = max(0, current_frame - window_size)
    end_idx = min(len(temporal_predictions), current_frame + window_size + 1)
    
    # Extraer predicciones de la ventana con ponderación por distancia
    label_votes = {}
    total_weight = 0.0
    
    for i in range(start_idx, end_idx):
        if i < len(temporal_predictions) and temporal_predictions[i] is not None:
            label, conf = temporal_predictions[i]
            
            # Ponderación: frames más cercanos tienen más peso
            distance = abs(i - current_frame)
            distance_weight = 1.0 / (1.0 + distance * 0.3)  # Decae con distancia
            
            # Peso combinado: distancia * confianza
            weight = distance_weight * conf
            
            if label not in label_votes:
                label_votes[label] = 0.0
            label_votes[label] += weight
            total_weight += weight
    
    if not label_votes or total_weight == 0:
        return None, 0.0
    
    # Mejor etiqueta
    best_label = max(label_votes, key=label_votes.get)
    
    # Confianza normalizada
    normalized_confidence = label_votes[best_label] / total_weight
    
    return best_label, normalized_confidence

# === PROCESAR VIDEO ===
print(f"\n📹 Abriendo video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ No se pudo abrir el video")
    sys.exit(1)

# Propiedades del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Leer metadatos de rotación (común en videos de celular)
rotation = 0
try:
    # Intentar leer metadatos de rotación
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
         '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1', VIDEO_PATH],
        capture_output=True, text=True, timeout=5
    )
    if result.stdout.strip():
        rotation = int(result.stdout.strip())
except:
    # Si no hay ffprobe o falla, intentar detectar por aspecto
    # Videos de celular vertical suelen ser 480x848 o 720x1280 pero guardados como 848x480
    pass

# Detectar si necesita rotación
needs_rotation = False
if rotation in [90, 270]:
    needs_rotation = True
    # Intercambiar dimensiones para la salida
    output_width, output_height = height, width
    is_vertical = True
else:
    output_width, output_height = width, height
    is_vertical = height > width

print(f"✅ Video abierto")
print(f"   Resolución archivo: {width}x{height}")
if needs_rotation:
    print(f"   Rotación detectada: {rotation}°")
    print(f"   Resolución real: {output_width}x{output_height} (VERTICAL)")
elif is_vertical:
    print(f"   Orientación: VERTICAL")
else:
    print(f"   Orientación: HORIZONTAL")
print(f"   FPS: {fps:.1f}")
print(f"   Total frames: {total_frames}")
print(f"   Duración: {total_frames/fps:.1f}s")

# Configurar VideoWriter con las dimensiones originales
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (output_width, output_height))

if not out.isOpened():
    print("❌ No se pudo crear el VideoWriter")
    cap.release()
    sys.exit(1)

frame_count = 0
prev_landmarks = None
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
temporal_predictions = []  # Guardar todas las predicciones para suavizado temporal
current_label = None
stability_counter = 0
predictions_log = []

print(f"\n🎬 Procesando video...")
print("   (Esto puede tomar unos minutos dependiendo del tamaño del video)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Aplicar rotación si es necesario (videos de celular)
        if needs_rotation:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Mostrar progreso cada 30 frames
        if frame_count % 30 == 0:
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
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            features = extract_features_from_landmarks(landmarks, prev_landmarks, fps)
            
            if features:
                label, confidence = predict_activity(features, model, scaler, label_encoder)
                
                # Guardar predicción original
                original_label = label
                original_confidence = confidence
                temporal_predictions.append((label, confidence))
                
                # Aplicar suavizado temporal para mejorar predicciones
                if confidence < 0.70:  # Mejorar predicciones con confianza media-baja
                    smoothed_label, smoothed_confidence = temporal_smoothing(
                        temporal_predictions, frame_count - 1, TEMPORAL_WINDOW
                    )
                    
                    # Usar suavizado si mejora significativamente
                    if smoothed_label and smoothed_confidence > confidence * 1.05:  # 5% mejor
                        label = smoothed_label
                        confidence = min(smoothed_confidence * 1.1, 0.99)  # Boost del 10%
                
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
                
                predictions_log.append({
                    'frame': frame_count,
                    'label': label,
                    'confidence': confidence,
                    'current': current_label,
                    'stability': stability_counter
                })
                
                # === DIBUJAR EN FRAME ===
                # Panel de fondo (reducido)
                cv2.rectangle(frame_display, (0, 0), (750, 120), (0, 0, 0), -1)
                cv2.rectangle(frame_display, (0, 0), (750, 120), (0, 255, 0), 2)
                
                # Frame número
                cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Actividad (GRANDE)
                color = (0, 255, 0) if stability_counter >= STABILITY_FRAMES else (0, 165, 255)
                cv2.putText(frame_display, f"ACTIVIDAD: {current_label or '?'}", (15, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Predicción actual
                cv2.putText(frame_display, f"Pred: {label} ({confidence:.0%})", (15, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Información adicional (opcional)
                # cv2.putText(frame_display, f"Buffer: {len(prediction_buffer)}", (15, 140),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            prev_landmarks = landmarks
        else:
            # No se detectó persona
            temporal_predictions.append(None)
            cv2.putText(frame_display, "No se detecta persona", (15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Escribir frame
        out.write(frame_display)

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido por el usuario")
except Exception as e:
    print(f"\n❌ Error durante procesamiento: {e}")
finally:
    cap.release()
    out.release()
    pose.close()
    
    # === GENERAR REPORTE MEJORADO ===
    print("\n" + "="*80)
    print("📊 REPORTE DETALLADO DE ANÁLISIS")
    print("="*80)
    
    if predictions_log:
        labels = [p['label'] for p in predictions_log]
        current_labels = [p['current'] for p in predictions_log if p['current']]
        confidences = [p['confidence'] for p in predictions_log]
        
        label_counts = Counter(labels)
        current_counts = Counter(current_labels)
        
        # Información del video
        print(f"\n📹 INFORMACIÓN DEL VIDEO:")
        print(f"   Archivo entrada:  {os.path.basename(VIDEO_PATH)}")
        print(f"   Archivo salida:   {OUTPUT_VIDEO_PATH}")
        print(f"   Resolución:       {output_width}x{output_height}")
        print(f"   Orientación:      {'VERTICAL' if is_vertical else 'HORIZONTAL'}")
        print(f"   Duración:         {total_frames/fps:.1f}s")
        print(f"   FPS:              {fps:.1f}")
        
        # Procesamiento
        print(f"\n⚙️  PROCESAMIENTO:")
        print(f"   Frames totales:   {total_frames}")
        print(f"   Frames procesados: {frame_count} ({100*frame_count/total_frames:.1f}%)")
        print(f"   Predicciones:     {len(predictions_log)}")
        frames_sin_persona = total_frames - len(predictions_log)
        if frames_sin_persona > 0:
            print(f"   Sin detección:    {frames_sin_persona} frames ({100*frames_sin_persona/total_frames:.1f}%)")
        
        # Estadísticas de confianza
        print(f"\n📊 MÉTRICAS DE CONFIANZA:")
        print(f"   Promedio:         {np.mean(confidences):.1%}")
        print(f"   Mínima:           {np.min(confidences):.1%}")
        print(f"   Máxima:           {np.max(confidences):.1%}")
        print(f"   Mediana:          {np.median(confidences):.1%}")
        
        # Clasificación de confianza
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.5)
        
        print(f"\n   Distribución:")
        print(f"   Alta (≥80%):      {high_conf:4d} frames ({100*high_conf/len(confidences):5.1f}%)")
        print(f"   Media (50-80%):   {med_conf:4d} frames ({100*med_conf/len(confidences):5.1f}%)")
        print(f"   Baja (<50%):      {low_conf:4d} frames ({100*low_conf/len(confidences):5.1f}%)")
        
        # Actividades detectadas (TODAS las predicciones)
        all_labels = [p['label'] for p in predictions_log]
        all_counts = Counter(all_labels)
        
        print(f"\n🎯 ACTIVIDADES DETECTADAS (todas las predicciones):")
        print(f"\n   {'Actividad':<20} {'Frames':>7} {'%':>7} {'Duración':>10} {'Gráfico'}")
        print(f"   {'-'*20} {'-'*7} {'-'*7} {'-'*10} {'-'*30}")
        for label, count in all_counts.most_common():
            pct = 100 * count / len(all_labels)
            duration = count / fps
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            print(f"   {label:<20} {count:7d} {pct:6.1f}% {duration:8.1f}s  {bar}")
        
        # Actividades estabilizadas
        if current_labels:
            print(f"\n🎯 ACTIVIDADES ESTABILIZADAS (confirmadas):")
            total_smooth = len(current_labels)
            for label, count in current_counts.most_common():
                pct = 100 * count / total_smooth
                duration = count / fps
                print(f"   {label:<20} {count:7d} frames ({pct:6.1f}%) - {duration:.1f}s")
        else:
            print(f"\n⚠️  Actividades estabilizadas: Ninguna")
            print(f"   (Las predicciones cambian muy rápido entre frames)")
        
        # Transiciones
        transitions = []
        for i in range(1, len(predictions_log)):
            if predictions_log[i]['current'] and predictions_log[i-1]['current']:
                if predictions_log[i]['current'] != predictions_log[i-1]['current']:
                    transitions.append({
                        'frame': predictions_log[i]['frame'],
                        'time': predictions_log[i]['frame'] / fps,
                        'from': predictions_log[i-1]['current'],
                        'to': predictions_log[i]['current']
                    })
        
        print(f"\n🔄 CAMBIOS DE ACTIVIDAD: {len(transitions)}")
        if transitions:
            print(f"\n   {'Tiempo':>8} {'Frame':>7}  {'Transición'}")
            print(f"   {'-'*8} {'-'*7}  {'-'*50}")
            for t in transitions[:15]:  # Mostrar primeros 15
                print(f"   {t['time']:7.1f}s {t['frame']:6d}  {t['from']:<20} → {t['to']:<20}")
            if len(transitions) > 15:
                print(f"   ... y {len(transitions)-15} cambios más")
        
        # Archivo de salida
        # Guardar log en JSON
        import json
        log_file = f"{video_name}_analysis_log.json"
        
        # Calcular promedio de confianza por actividad
        confidence_by_activity = {}
        for p in predictions_log:
            label = p['label']
            conf = p['confidence']
            if label not in confidence_by_activity:
                confidence_by_activity[label] = []
            confidence_by_activity[label].append(conf)
        
        # Calcular promedios
        avg_confidence_by_activity = {}
        for label, confs in confidence_by_activity.items():
            avg_confidence_by_activity[label] = {
                'average': float(np.mean(confs)),
                'min': float(np.min(confs)),
                'max': float(np.max(confs)),
                'median': float(np.median(confs)),
                'count': len(confs)
            }
        
        log_data = {
            'video': VIDEO_PATH,
            'total_frames': total_frames,
            'fps': float(fps),
            'resolution': f"{output_width}x{output_height}",
            'duration': f"{total_frames/fps:.1f}s",
            'predictions_count': len(predictions_log),
            'transitions': len(transitions),
            'confidence_avg': float(np.mean(confidences)),
            'confidence_min': float(np.min(confidences)),
            'confidence_max': float(np.max(confidences)),
            'confidence_median': float(np.median(confidences)),
            'activities_detected': {label: int(count) for label, count in all_counts.items()},
            'confidence_by_activity': avg_confidence_by_activity,
            'predictions': []
        }
        
        # Agregar todas las predicciones
        for p in predictions_log:
            log_data['predictions'].append({
                'frame': int(p['frame']),
                'label': p['label'],
                'confidence': float(p['confidence']),
                'current': p['current'],
                'stability': int(p['stability'])
            })
        
        # Guardar JSON
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ARCHIVOS GENERADOS:")
        print(f"   Video: {OUTPUT_VIDEO_PATH}")
        if os.path.exists(OUTPUT_VIDEO_PATH):
            size_mb = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024*1024)
            print(f"   Tamaño: {size_mb:.1f} MB")
        print(f"   Log JSON: {log_file}")
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            print(f"   Tamaño: {size_kb:.1f} KB")
        
        # Recomendaciones
        avg_conf = np.mean(confidences)
        print(f"\n💡 RECOMENDACIONES:")
        if avg_conf < 0.5:
            print("   ⚠️  Confianza baja (<50%):")
            print("      - Asegúrate de que la persona esté completamente visible")
            print("      - Mejora la iluminación del video")
            print("      - Evita que la persona esté muy cerca o muy lejos de la cámara")
        elif avg_conf < 0.7:
            print("   ℹ️  Confianza media (50-70%):")
            print("      - El modelo funciona pero puede mejorar")
            print("      - Intenta con mejor iluminación y encuadre")
        else:
            print("   ✅ Confianza buena (≥70%):")
            print("      - El modelo está funcionando correctamente")
            print("      - Las predicciones son confiables")
        
        if frames_sin_persona > total_frames * 0.1:
            print(f"   ⚠️  {100*frames_sin_persona/total_frames:.0f}% de frames sin persona detectada")
            print("      - Verifica que la persona esté en el encuadre todo el tiempo")
    
    print("\n" + "="*80)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n🎬 Abre el video: {OUTPUT_VIDEO_PATH}")
    print("   Para ver las predicciones en tiempo real con el skeleton dibujado")
