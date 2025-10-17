"""
Script de verificación del pipeline
Comprueba que todo está configurado correctamente antes de ejecutar
"""

import json
import os
import sys

print("=" * 70)
print("🔍 VERIFICACIÓN DEL PIPELINE")
print("=" * 70)

errors = []
warnings = []

# 1. Verificar archivos de entrada
print("\n1️⃣  Archivos de entrada...")

if not os.path.exists("project-label-studio.json"):
    errors.append("❌ No encontrado: project-label-studio.json")
else:
    try:
        with open("project-label-studio.json", "r") as f:
            label_data = json.load(f)
        n_videos = len(label_data)
        print(f"   ✓ project-label-studio.json ({n_videos} videos etiquetados)")
    except json.JSONDecodeError:
        errors.append("❌ project-label-studio.json no es JSON válido")

if not os.path.exists("Videos APO"):
    errors.append("❌ No encontrado: carpeta Videos APO/")
else:
    videos = [f for f in os.listdir("Videos APO") if f.endswith(".mp4")]
    print(f"   ✓ Videos APO/ ({len(videos)} archivos .mp4)")
    if len(videos) == 0:
        warnings.append("⚠️  Carpeta Videos APO/ vacía")

# 2. Verificar scripts
print("\n2️⃣  Scripts requeridos...")

scripts = [
    "extract_mediapipe_data.py",
    "enrich_dataset.py",
    "eda_basic.py"
]

for script in scripts:
    if os.path.exists(script):
        print(f"   ✓ {script}")
    else:
        errors.append(f"❌ No encontrado: {script}")

# 3. Verificar archivos de salida del paso 1
print("\n3️⃣  Salida del paso 1 (extract_mediapipe_data.py)...")

if os.path.exists("mediapipe_labels_dataset.csv"):
    try:
        import pandas as pd
        df = pd.read_csv("mediapipe_labels_dataset.csv")
        n_frames = len(df)
        n_cols = len(df.columns)
        print(f"   ✓ mediapipe_labels_dataset.csv ({n_frames} frames, {n_cols} columnas)")
        
        # Verificar columnas esperadas
        expected_cols = [
            "video_id", "frame_opencv", "frame_labelstudio",
            "fps", "timestamp_ms", "width", "height",
            "mean_visibility", "num_visible_lms",
            "hip_center_x", "hip_center_y", "torso_scale",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "bbox_area", "bbox_aspect",
            "label"
        ]
        
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            warnings.append(f"⚠️  Columnas faltantes: {missing_cols}")
        
        # Verificar landmarks
        landmark_cols = [c for c in df.columns if c.startswith("x_")]
        if len(landmark_cols) == 33:
            print(f"   ✓ Landmarks (33 puntos × 4 coords = {33*4} columnas)")
        else:
            warnings.append(f"⚠️  Landmarks incompletos: {len(landmark_cols)} puntos")
            
    except Exception as e:
        errors.append(f"❌ Error al leer mediapipe_labels_dataset.csv: {e}")
else:
    print("   ⓘ mediapipe_labels_dataset.csv no generado aún (ejecuta paso 1)")

# 4. Verificar archivos de salida del paso 2
print("\n4️⃣  Salida del paso 2 (enrich_dataset.py)...")

if os.path.exists("mediapipe_labels_dataset_enriched.csv"):
    try:
        import pandas as pd
        df = pd.read_csv("mediapipe_labels_dataset_enriched.csv")
        n_frames = len(df)
        n_cols = len(df.columns)
        print(f"   ✓ mediapipe_labels_dataset_enriched.csv ({n_frames} frames, {n_cols} columnas)")
        
        # Verificar features derivados
        derived_cols = [
            "speed_15", "speed_16", "speed_25", "speed_26", "speed_27", "speed_28",
            "knee_left_deg", "knee_right_deg", "elbow_left_deg", "elbow_right_deg",
            "segment_id", "low_quality"
        ]
        
        missing_derived = [c for c in derived_cols if c not in df.columns]
        if missing_derived:
            warnings.append(f"⚠️  Columnas derivadas faltantes: {missing_derived}")
        else:
            print(f"   ✓ Features derivados (velocidades, ángulos, segmentación)")
            
    except Exception as e:
        errors.append(f"❌ Error al leer mediapipe_labels_dataset_enriched.csv: {e}")
else:
    print("   ⓘ mediapipe_labels_dataset_enriched.csv no generado aún (ejecuta paso 2)")

# 5. Verificar salida del paso 3 (gráficos)
print("\n5️⃣  Salida del paso 3 (eda_basic.py)...")

eda_figures = [
    "eda_01_label_distribution.png",
    "eda_02_frames_per_video.png",
    "eda_03_landmark_quality.png",
    "eda_04_velocities.png",
    "eda_05_angles.png",
    "eda_06_angles_by_label.png",
    "eda_07_segment_duration.png",
]

found_figures = sum(1 for f in eda_figures if os.path.exists(f))
if found_figures > 0:
    print(f"   ✓ {found_figures}/{len(eda_figures)} gráficos generados")
else:
    print("   ⓘ Gráficos no generados aún (ejecuta paso 3)")

# 6. Verificar dependencias
print("\n6️⃣  Dependencias Python...")

required_packages = [
    ("cv2", "OpenCV"),
    ("mediapipe", "MediaPipe"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("tqdm", "tqdm"),
]

all_packages_ok = True
for module, name in required_packages:
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        errors.append(f"❌ No instalado: {name} (pip install {module})")
        all_packages_ok = False

# Resumen
print("\n" + "=" * 70)
print("📋 RESUMEN")
print("=" * 70)

if errors:
    print(f"\n❌ ERRORES ({len(errors)}):")
    for error in errors:
        print(f"   {error}")

if warnings:
    print(f"\n⚠️  ADVERTENCIAS ({len(warnings)}):")
    for warning in warnings:
        print(f"   {warning}")

if not errors:
    print("\n✅ TODO CORRECTO - Listo para ejecutar el pipeline\n")
    print("Próximos pasos:")
    print("   1. python3 extract_mediapipe_data.py")
    print("   2. python3 enrich_dataset.py")
    print("   3. python3 eda_basic.py")
else:
    print("\n⚠️  Hay errores que deben corregirse antes de continuar")
    sys.exit(1)
