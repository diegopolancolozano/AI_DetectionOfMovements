# Entrega 2 - Preparación y Modelación

Este directorio contiene el código y datos para la segunda entrega del proyecto de detección de movimientos humanos.

## Estructura de Archivos

### Datos
- `mediapipe_labels_dataset_combined.csv` - Dataset combinado con todos los videos (Entrega 1 + Entrega 2)
- `mediapipe_labels_dataset_combined_enriched.csv` - Dataset enriquecido con features derivados
- `videos familiaDiego/` - Videos y anotaciones del grupo familia Diego (5 videos)
- `videos otroGrupo/` - Videos y anotaciones de otro grupo (20 videos)

### Scripts
- `process_all_videos.py` - Script principal para procesar todos los videos y generar datasets combinados

### Notebooks
- `PreparacionYModelacion.ipynb` - Notebook principal con preparación de datos, entrenamiento y evaluación de modelos

### Modelos Entrenados
- `models/` - Directorio con los modelos guardados (.joblib)
  - `best_model.joblib` - Mejor modelo seleccionado
  - `rf_model.joblib` - Random Forest
  - `xgb_model.joblib` - XGBoost
  - `svm_model.joblib` - SVM
  - `scaler.joblib` - Escalador de features
  - `label_encoder.joblib` - Codificador de etiquetas

## Cómo Usar

### 1. Procesar Videos y Generar Dataset Combinado

Si agregas nuevos videos, ejecuta el script de procesamiento:

```bash
cd "Entrega 2"
python process_all_videos.py
```

Este script:
- Procesa videos de Entrega 1 (Videos APO)
- Procesa videos de Entrega 2 (familia Diego + otro grupo)
- Extrae landmarks con MediaPipe Pose
- Combina todos los datos en un archivo CSV
- Enriquece el dataset con features derivados (velocidades, ángulos, etc.)

### 2. Entrenar Modelos

Abre y ejecuta el notebook:

```bash
jupyter notebook PreparacionYModelacion.ipynb
```

El notebook incluye:
- Carga y exploración del dataset combinado
- Limpieza y preparación de datos
- Entrenamiento de modelos (Random Forest, XGBoost, SVM)
- Evaluación y comparación de modelos
- Guardado de modelos entrenados

## Dataset Combinado

El dataset combinado incluye:

### Fuentes de Datos
- **Entrega1**: 18 videos originales
- **Entrega2_FamiliaDiego**: 5 videos nuevos
- **Entrega2_OtroGrupo**: 20 videos nuevos

### Total
- Aproximadamente 43 videos
- ~30,000+ frames
- 8 actividades clasificadas

### Actividades
1. Standing (De pie)
2. Sitting (Sentado)
3. Walk forward (Caminar hacia adelante)
4. Walk backward (Caminar hacia atrás)
5. Turn (Girar)
6. Sit down (Sentarse)
7. Get up (Levantarse)
8. Unlabeled (Sin etiquetar)

### Features

El dataset enriquecido contiene:

#### Metadata
- `video_id` - ID del video
- `source` - Fuente de datos (Entrega1, Entrega2_FamiliaDiego, Entrega2_OtroGrupo)
- `frame_opencv` - Número de frame en OpenCV
- `frame_labelstudio` - Número de frame en Label Studio
- `fps` - Frames por segundo
- `timestamp_ms` - Timestamp en milisegundos
- `width`, `height` - Dimensiones del video

#### Landmarks (33 puntos × 4 valores)
- `x_0` a `x_32` - Coordenadas X normalizadas
- `y_0` a `y_32` - Coordenadas Y normalizadas
- `z_0` a `z_32` - Coordenadas Z (profundidad)
- `v_0` a `v_32` - Visibilidad/confianza

#### Calidad
- `mean_visibility` - Visibilidad promedio de landmarks
- `num_visible_lms` - Número de landmarks visibles
- `low_quality` - Marca de baja calidad

#### Posición y Escala
- `hip_center_x`, `hip_center_y` - Centro de caderas
- `torso_scale` - Escala del torso
- `bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax` - Bounding box
- `bbox_area`, `bbox_aspect` - Área y aspecto del bounding box

#### Features Derivados
- `speed_15` a `speed_28` - Velocidades de muñecas, rodillas y tobillos
- `knee_left_deg`, `knee_right_deg` - Ángulos de rodillas
- `elbow_left_deg`, `elbow_right_deg` - Ángulos de codos
- `segment_id` - ID de segmento temporal
- `fps_eff` - FPS efectivo

#### Etiqueta
- `label` - Actividad clasificada

## Notas

- Los videos (.mp4) están ignorados en `.gitignore` para no subir archivos grandes al repositorio
- Los archivos CSV y JSON de anotaciones se mantienen en el repositorio
- El dataset combinado se regenera cada vez que se ejecuta `process_all_videos.py`
