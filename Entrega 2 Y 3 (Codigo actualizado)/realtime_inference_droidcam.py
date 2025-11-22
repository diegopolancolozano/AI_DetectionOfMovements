"""
Sistema de Inferencia en Tiempo Real - OPTIMIZADO PARA DROIDCAM
Detecta movimientos humanos usando MediaPipe y modelo pre-entrenado
"""

import math
import warnings
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

# Suprimir warnings de scikit-learn
warnings.filterwarnings('ignore', category=UserWarning)

# === CONFIGURACIÓN ===
MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"

# Buffer para suavizado de predicciones (última N predicciones)
PREDICTION_BUFFER_SIZE = 10  # Aumentado para mejor estabilidad

# CONFIGURACIÓN PARA DROIDCAM
# Cambia estos valores si es necesario
CAMERA_INDEX = 0  #  DROIDCAM DETECTADO EN ÍNDICE 2
DROIDCAM_WIDTH = 1920  # Resolución reducida para mejor rendimiento
DROIDCAM_HEIGHT = 1080
SKIP_FRAMES = 1  # Procesar 1 de cada N frames (1 = todos, 2 = la mitad, etc.)
ROTATE_FRAME = True  # ✅ Rotar 90 grados para Droidcam
ROTATION_ANGLE = 90  # Ángulo de rotación en grados (90, 180, 270)

# === CARGAR MODELO Y TRANSFORMADORES ===
print("🔄 Cargando modelo y transformadores...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Modelo cargado exitosamente")
    print(f"   Clases: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit(1)

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=lite, 1=full, 2=heavy - usa 0 para más velocidad
    smooth_landmarks=True,  # Suavizado de landmarks para mejor estabilidad
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === FUNCIÓN DE COMBINACIÓN DE CLASES ===
def combine_classes(label):
    """Combina clases similares para simplificar el modelo"""
    if label in ["Walk forward", "Walk backward"]:
        return "Walking"
    elif label in ["Get up", "Sit down"]:
        return "Transition"
    else:
        return label

# === FUNCIONES DE EXTRACCIÓN DE FEATURES ===

def angle_deg(ax, ay, bx, by, cx, cy):
    """Calcula el ángulo en B del triángulo A-B-C en grados."""
    BAx, BAy = ax - bx, ay - by
    BCx, BCy = cx - bx, cy - by
    num = BAx * BCx + BAy * BCy
    den = math.sqrt(BAx**2 + BAy**2) * math.sqrt(BCx**2 + BCy**2)
    if den == 0:
        return np.nan
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def extract_features_from_landmarks(landmarks, prev_landmarks=None, fps=30.0):
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


def predict_activity(features_dict, model, scaler, label_encoder):
    """
    Realiza predicción usando el modelo entrenado.
    """
    # Convertir a DataFrame
    df = pd.DataFrame([features_dict])
    
    # Seleccionar solo las features que el modelo espera
    expected_features = scaler.feature_names_in_
    df_filtered = df[expected_features]
    df_filtered = df_filtered.fillna(0)
    
    # Escalar y predecir
    try:
        X_scaled = scaler.transform(df_filtered)
        
        # Convertir de vuelta a DataFrame con nombres de columnas
        X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)
        
        y_pred = model.predict(X_scaled_df)
        y_proba = model.predict_proba(X_scaled_df)
        
        label = label_encoder.inverse_transform(y_pred)[0]
        confidence = np.max(y_proba)
        
        return label, confidence
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return "Error", 0.0


def rotate_frame(frame, angle):
    """
    Rota un frame el ángulo especificado en grados.
    
    Args:
        frame: Imagen de OpenCV
        angle: Ángulo de rotación (90, 180, 270)
    
    Returns:
        Frame rotado
    """
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        return frame


# === MAIN LOOP ===
def main():
    print("\n" + "="*70)
    print("🎥 SISTEMA DE DETECCIÓN DE MOVIMIENTOS - OPTIMIZADO PARA DROIDCAM")
    print("="*70)
    print(f"📱 Configuración Droidcam:")
    print(f"   - Índice de cámara: {CAMERA_INDEX}")
    print(f"   - Resolución: {DROIDCAM_WIDTH}x{DROIDCAM_HEIGHT}")
    print(f"   - Skip frames: {SKIP_FRAMES}")
    print("\n💡 Presiona 'Q' para salir")
    print("="*70 + "\n")
    
    # Intentar abrir la cámara
    print(f"📹 Intentando abrir cámara índice {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara con índice {CAMERA_INDEX}")
        print("\n💡 Soluciones:")
        print("   1. Abre Droidcam en tu celular")
        print("   2. Inicia DroidCam Client en tu PC")
        print("   3. Si no funciona, cambia CAMERA_INDEX en el código:")
        print("      - Prueba con 0, 1, o 2")
        print("   4. O usa la URL directa: cv2.VideoCapture('http://192.168.x.x:4747/video')")
        return
    
    # Configurar resolución y opciones para Droidcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DROIDCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DROIDCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir latencia
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"✅ Cámara abierta correctamente")
    print(f"   📏 Resolución: {actual_width}x{actual_height}")
    print(f"   🎬 FPS: {fps:.1f}")
    if ROTATE_FRAME:
        print(f"   🔄 Rotación: {ROTATION_ANGLE}° (ACTIVADA)")
    print()
    
    # Crear ventana UNA SOLA VEZ
    window_name = 'Deteccion de Movimientos [Q para salir]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    prev_landmarks = None
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    frame_count = 0
    process_count = 0
    
    print("🚀 Procesando... (puede tomar unos segundos en el primer frame)\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al leer frame")
                break
            
            frame_count += 1
            
            # Aplicar rotación si está habilitada
            if ROTATE_FRAME:
                frame = rotate_frame(frame, ROTATION_ANGLE)
            
            # Procesar solo algunos frames para mejor rendimiento
            should_process = (frame_count % SKIP_FRAMES == 0)
            
            if should_process:
                process_count += 1
                
                # Convertir a RGB para MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Procesar landmarks si existen
                if results.pose_landmarks:
                    # Dibujar landmarks
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Extraer features
                    landmarks = results.pose_landmarks.landmark
                    features = extract_features_from_landmarks(landmarks, prev_landmarks, fps)
                    
                    # Predecir si tenemos features válidas
                    if features:
                        label, confidence = predict_activity(features, model, scaler, label_encoder)
                        
                        # Combinar clases (si el modelo predice clases originales)
                        label = combine_classes(label)
                        
                        # Agregar al buffer para suavizado
                        prediction_buffer.append(label)
                        
                        # Usar la predicción más común del buffer
                        if len(prediction_buffer) > 0:
                            from collections import Counter
                            most_common = Counter(prediction_buffer).most_common(1)[0]
                            smoothed_label = most_common[0]
                        else:
                            smoothed_label = label
                        
                        # Mostrar predicción en el frame con fondo
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (5, 5), (450, 125), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        
                        cv2.putText(
                            frame,
                            f"Actividad: {smoothed_label}",
                            (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            frame,
                            f"Confianza: {confidence:.1%}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )
                        
                        # Debug: buffer de predicciones
                        buffer_text = ", ".join(list(prediction_buffer)[-3:])
                        cv2.putText(
                            frame,
                            f"Buffer: {buffer_text}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Features insuficientes",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                    
                    # Guardar landmarks para el siguiente frame
                    prev_landmarks = landmarks
                else:
                    cv2.putText(
                        frame,
                        "No se detecta persona - Alejate un poco",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    prev_landmarks = None
                    prediction_buffer.clear()
            
            # Mostrar contador (siempre)
            cv2.putText(
                frame,
                f"Frames: {process_count}",
                (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Mostrar instrucciones
            cv2.putText(
                frame,
                "Presiona Q para salir",
                (frame.shape[1] - 250, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Mostrar frame en LA MISMA VENTANA
            cv2.imshow(window_name, frame)
            
            # Salir con 'q' o 'Q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Detenido por el usuario (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n✅ Sesión finalizada")
        print(f"📊 Estadísticas: {process_count} frames procesados de {frame_count} capturados")


if __name__ == "__main__":
    main()
