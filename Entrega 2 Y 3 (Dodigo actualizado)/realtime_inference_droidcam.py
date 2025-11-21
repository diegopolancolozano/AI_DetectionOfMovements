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
PREDICTION_BUFFER_SIZE = 5

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
    Extrae todas las características necesarias desde los landmarks de MediaPipe.
    """
    features = {}
    
    # Extraer coordenadas básicas
    xs, ys, vis_vals = [], [], []
    for i, lm in enumerate(landmarks):
        xs.append(lm.x)
        ys.append(lm.y)
        vis_vals.append(lm.visibility)
    
    # === FEATURES DE CALIDAD ===
    features['mean_visibility'] = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0
    features['num_visible_lms'] = int(sum(1 for v in vis_vals if v >= 0.5))
    
    # === BOUNDING BOX (requerido por el modelo) ===
    if xs and ys:
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        features['bbox_xmin'] = xmin
        features['bbox_ymin'] = ymin
        features['bbox_xmax'] = xmax
        features['bbox_ymax'] = ymax
        features['bbox_area'] = max((xmax - xmin), 0.0) * max((ymax - ymin), 0.0)
        features['bbox_aspect'] = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else None
    else:
        features['bbox_xmin'] = features['bbox_ymin'] = None
        features['bbox_xmax'] = features['bbox_ymax'] = None
        features['bbox_area'] = features['bbox_aspect'] = None
    
    # === VELOCIDADES (requiere frame anterior) ===
    keys = [15, 16, 25, 26, 27, 28]  # muñecas, rodillas, tobillos
    for i in keys:
        if prev_landmarks is not None and i < len(landmarks) and i < len(prev_landmarks):
            dx = landmarks[i].x - prev_landmarks[i].x
            dy = landmarks[i].y - prev_landmarks[i].y
            features[f'speed_{i}'] = math.sqrt(dx**2 + dy**2) * fps
        else:
            features[f'speed_{i}'] = 0.0
    
    # === ÁNGULOS DE ARTICULACIONES ===
    # Rodillas
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
    
    # Codos
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
                    
                    # Predecir solo si tenemos suficiente calidad
                    if features['mean_visibility'] > 0.5 and features['num_visible_lms'] >= 20:
                        label, confidence = predict_activity(features, model, scaler, label_encoder)
                        
                        # Agregar al buffer para suavizado
                        prediction_buffer.append(label)
                        
                        # Usar la predicción más común del buffer
                        if len(prediction_buffer) > 0:
                            from collections import Counter
                            most_common = Counter(prediction_buffer).most_common(1)[0]
                            smoothed_label = most_common[0]
                        else:
                            smoothed_label = label
                        
                        # Mostrar predicción en el frame
                        cv2.putText(
                            frame,
                            f"Actividad: {smoothed_label}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            frame,
                            f"Confianza: {confidence:.1%}",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                        
                        # Mostrar calidad
                        cv2.putText(
                            frame,
                            f"Landmarks: {features['num_visible_lms']}/33",
                            (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Calidad insuficiente - Mejora iluminacion",
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
