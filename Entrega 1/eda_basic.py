"""
EDA básico para el dataset de MediaPipe + Labels
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# Cargar dataset enriquecido
print("📂 Cargando dataset enriquecido...")
df = pd.read_csv("mediapipe_labels_dataset_enriched.csv")
print(f"✓ {len(df)} frames, {df.shape[1]} columnas\n")

# 1. DISTRIBUCIÓN DE ETIQUETAS
print("=" * 60)
print("1. DISTRIBUCIÓN DE ETIQUETAS")
print("=" * 60)
label_counts = df["label"].value_counts()
print(label_counts)
print(f"\nBalance (%): \n{(100 * label_counts / len(df)).round(2)}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
label_counts.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Conteo de frames por etiqueta")
axes[0].set_ylabel("Frames")

(100 * label_counts / len(df)).plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Distribución de etiquetas (%)")
axes[1].set_ylabel("Porcentaje (%)")
plt.tight_layout()
plt.savefig("eda_01_label_distribution.png", dpi=100, bbox_inches="tight")
print("✓ Gráfico guardado: eda_01_label_distribution.png\n")

# 2. FRAMES POR VIDEO
print("=" * 60)
print("2. FRAMES POR VIDEO")
print("=" * 60)
frames_per_video = df.groupby("video_id")["frame_opencv"].nunique()
print(frames_per_video.describe())
print(f"\nTotal videos: {len(frames_per_video)}\n")

fig, ax = plt.subplots(figsize=(12, 5))
frames_per_video.plot(kind="bar", ax=ax, color="mediumseagreen")
ax.set_title("Frames por video")
ax.set_ylabel("Número de frames")
ax.set_xlabel("Video ID")
plt.tight_layout()
plt.savefig("eda_02_frames_per_video.png", dpi=100, bbox_inches="tight")
print("✓ Gráfico guardado: eda_02_frames_per_video.png\n")

# 3. CALIDAD DE LANDMARKS
print("=" * 60)
print("3. CALIDAD DE LANDMARKS")
print("=" * 60)
if "mean_visibility" in df:
    print(f"Mean visibility: \n{df['mean_visibility'].describe()}\n")
    print(f"Num visible landmarks: \n{df['num_visible_lms'].describe()}\n")
    
    if "low_quality" in df:
        n_low = df["low_quality"].sum()
        print(f"Frames de baja calidad: {n_low} ({100*n_low/len(df):.2f}%)\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["mean_visibility"], bins=50, color="skyblue", edgecolor="black")
    axes[0].set_title("Distribución de mean_visibility")
    axes[0].set_xlabel("Mean visibility")
    axes[0].set_ylabel("Frames")
    
    axes[1].hist(df["num_visible_lms"], bins=34, color="lightcoral", edgecolor="black")
    axes[1].set_title("Distribución de landmarks visibles")
    axes[1].set_xlabel("Número de landmarks visibles")
    axes[1].set_ylabel("Frames")
    plt.tight_layout()
    plt.savefig("eda_03_landmark_quality.png", dpi=100, bbox_inches="tight")
    print("✓ Gráfico guardado: eda_03_landmark_quality.png\n")

# 4. RANGO DE COORDENADAS
print("=" * 60)
print("4. RANGO DE COORDENADAS")
print("=" * 60)
x_cols = [c for c in df.columns if c.startswith("x_") and not c.startswith("x_bbox")]
y_cols = [c for c in df.columns if c.startswith("y_") and not c.startswith("y_bbox")]
print(f"X: [{df[x_cols].min().min():.4f}, {df[x_cols].max().max():.4f}]")
print(f"Y: [{df[y_cols].min().min():.4f}, {df[y_cols].max().max():.4f}]")
print()

# 5. VELOCIDADES
print("=" * 60)
print("5. VELOCIDADES POR LANDMARK")
print("=" * 60)
speed_cols = [c for c in df.columns if c.startswith("speed_")]
if speed_cols:
    for col in speed_cols:
        print(f"{col}: {df[col].describe().to_dict()}")
    print()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, col in enumerate(speed_cols[:6]):
        axes[idx].hist(df[col].dropna(), bins=50, color="mediumpurple", edgecolor="black")
        axes[idx].set_title(f"Distribución de {col}")
        axes[idx].set_xlabel("Velocidad (normalized/seg)")
    plt.tight_layout()
    plt.savefig("eda_04_velocities.png", dpi=100, bbox_inches="tight")
    print("✓ Gráfico guardado: eda_04_velocities.png\n")

# 6. ÁNGULOS
print("=" * 60)
print("6. ÁNGULOS DE ARTICULACIONES")
print("=" * 60)
angle_cols = [c for c in df.columns if "deg" in c]
if angle_cols:
    for col in angle_cols:
        print(f"{col}: {df[col].describe().to_dict()}")
    print()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(angle_cols):
        axes[idx].hist(df[col].dropna(), bins=50, color="lightseagreen", edgecolor="black")
        axes[idx].set_title(f"Distribución de {col}")
        axes[idx].set_xlabel("Ángulo (grados)")
    plt.tight_layout()
    plt.savefig("eda_05_angles.png", dpi=100, bbox_inches="tight")
    print("✓ Gráfico guardado: eda_05_angles.png\n")

# 7. BOX PLOTS POR ETIQUETA
print("=" * 60)
print("7. ÁNGULOS POR ETIQUETA (BOX PLOT)")
print("=" * 60)
if angle_cols:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(angle_cols):
        df.boxplot(column=col, by="label", ax=axes[idx])
        axes[idx].set_title(f"{col} por etiqueta")
        axes[idx].set_xlabel("Etiqueta")
        axes[idx].set_ylabel(col)
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("eda_06_angles_by_label.png", dpi=100, bbox_inches="tight")
    print("✓ Gráfico guardado: eda_06_angles_by_label.png\n")

# 8. SEGMENTACIÓN TEMPORAL
print("=" * 60)
print("8. ANÁLISIS DE SEGMENTACIÓN TEMPORAL")
print("=" * 60)
if "segment_id" in df:
    seg_info = df.groupby(["video_id", "segment_id", "label"]).size().reset_index(name="duration")
    print(f"Total segmentos: {len(seg_info)}")
    print(f"\nDuración por etiqueta:")
    print(seg_info.groupby("label")["duration"].describe())
    print()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    seg_info.boxplot(column="duration", by="label", ax=ax)
    ax.set_title("Duración de segmentos por etiqueta")
    ax.set_xlabel("Etiqueta")
    ax.set_ylabel("Duración (frames)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("eda_07_segment_duration.png", dpi=100, bbox_inches="tight")
    print("✓ Gráfico guardado: eda_07_segment_duration.png\n")

print("=" * 60)
print("✅ EDA COMPLETADO")
print("=" * 60)
print("Archivos generados:")
print("  - eda_01_label_distribution.png")
print("  - eda_02_frames_per_video.png")
print("  - eda_03_landmark_quality.png")
print("  - eda_04_velocities.png")
print("  - eda_05_angles.png")
print("  - eda_06_angles_by_label.png")
print("  - eda_07_segment_duration.png")
