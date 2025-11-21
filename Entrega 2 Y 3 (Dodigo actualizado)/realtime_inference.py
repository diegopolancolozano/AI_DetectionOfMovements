"""
Sistema de Inferencia en Tiempo Real
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
MODEL_PATH = "models/xgb_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"

# Buffer para suavizado de predicciones (última N predicciones)
PREDICTION_BUFFER_SIZE = 5

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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    # Convertir a DataFrame (necesario para que coincidan los nombres de columnas)
    df = pd.DataFrame([features_dict])
    
    # Seleccionar solo las features que el modelo espera
    expected_features = scaler.feature_names_in_
    
    # Filtrar y ordenar según lo que espera el modelo
    df_filtered = df[expected_features]
    
    # Manejar valores faltantes
    df_filtered = df_filtered.fillna(0)
    
    # Escalar features
    try:
        X_scaled = scaler.transform(df_filtered)
        
        # Convertir de vuelta a DataFrame con nombres de columnas para evitar warnings
        X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)
        
        y_pred = model.predict(X_scaled_df)
        y_proba = model.predict_proba(X_scaled_df)
        
        label = label_encoder.inverse_transform(y_pred)[0]
        confidence = np.max(y_proba)
        
        return label, confidence
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return "Error", 0.0


# === MAIN LOOP ===
def main():
    print("\n" + "="*60)
    print("🎥 SISTEMA DE DETECCIÓN DE MOVIMIENTOS EN TIEMPO REAL")
    print("="*60)
    print("Presiona 'q' para salir")
    print("="*60 + "\n")
    
    # Intentar abrir la cámara
    # Para Droidcam, usa el índice 1 o 2 si 0 no funciona
    camera_index = 2
    cap = cv2.VideoCapture(camera_index)
    
    # Configuraciones para mejor rendimiento con Droidcam
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menos latencia
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        print("💡 Si usas Droidcam, intenta cambiar el índice de cámara")
        print("   Abre el código y cambia VideoCapture(0) a VideoCapture(1) o VideoCapture(2)")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"📹 FPS de la cámara: {fps:.1f}")
    print(f"📏 Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Crear ventana una sola vez antes del bucle
    window_name = 'Deteccion de Movimientos - Presiona Q para salir'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    prev_landmarks = None
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer frame")
            break
        
        # Rotar 90 grados a la derecha si es la cámara 2
        if camera_index == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Dibujar landmarks
        if results.pose_landmarks:
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
                
                # Agregar al buffer para suavizado
                prediction_buffer.append(label)
                
                # Usar la predicción más común del buffer
                if len(prediction_buffer) > 0:
                    from collections import Counter
                    most_common = Counter(prediction_buffer).most_common(1)[0]
                    smoothed_label = most_common[0]
                    smoothed_confidence = most_common[1] / len(prediction_buffer)
                else:
                    smoothed_label = label
                    smoothed_confidence = confidence
                
                # Mostrar predicción en el frame
                cv2.putText(
                    frame,
                    f"Actividad: {smoothed_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Confianza: {confidence:.2%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Calidad insuficiente",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            
            # Guardar landmarks para el siguiente frame
            prev_landmarks = landmarks
        else:
            cv2.putText(
                frame,
                "No se detecta persona",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            prev_landmarks = None
            prediction_buffer.clear()
        
        # Mostrar FPS real
        frame_count += 1
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Mostrar frame
        cv2.imshow(window_name, frame)
        
        # Salir con 'q' - aumentado el tiempo de espera para mejor rendimiento
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\n✅ Sesión finalizada")


if __name__ == "__main__":
    main()
