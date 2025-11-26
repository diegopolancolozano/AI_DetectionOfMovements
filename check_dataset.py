import pandas as pd

# Leer el CSV generado
df = pd.read_csv('Entrega 2 Y 3 (Dodigo actualizado)/mediapipe_labels_dataset_combined_enriched.csv')

print("="*70)
print("DATASET GENERADO POR process_all_videos.py")
print("="*70)

print(f"\n📊 Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")

print(f"\n📁 Fuentes de datos:")
if 'source' in df.columns:
    print(df['source'].value_counts())

print(f"\n🎬 Videos procesados: {df['video_id'].nunique()}")

print(f"\n🏷️ Distribución de etiquetas:")
print(df['label'].value_counts())

print(f"\n📋 Primeras 15 columnas:")
print(list(df.columns[:15]))

print(f"\n📈 Estadísticas de calidad:")
if 'mean_visibility' in df.columns:
    print(f"   Visibilidad promedio: {df['mean_visibility'].mean():.2%}")
if 'num_visible_lms' in df.columns:
    print(f"   Landmarks visibles (promedio): {df['num_visible_lms'].mean():.1f}/33")

print(f"\n✅ Este dataset se usó para entrenar los modelos en models/")
print("="*70)
