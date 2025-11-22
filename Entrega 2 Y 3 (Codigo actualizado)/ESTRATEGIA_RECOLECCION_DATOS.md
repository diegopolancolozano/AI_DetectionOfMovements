# Estrategia de Recolección de Datos - Entrega 2

## Motivación

Para la segunda entrega del proyecto, identificamos que el dataset original (Entrega 1) presentaba limitaciones:

1. **Volumen limitado**: Solo 18 videos (~10,380 frames)
2. **Poca diversidad demográfica**: Videos grabados por el mismo grupo
3. **Posible sobreajuste**: Los modelos podrían aprender características específicas de los participantes originales en lugar de patrones generales de movimiento
4. **Sesgo postural**: En los videos originales, las personas mantenían los brazos en posiciones similares durante "Standing" y otras actividades

## Estrategia Implementada

### 1. Grabaciones con Familiares (Familia Diego)

**Objetivo**: Aumentar diversidad demográfica y variabilidad en la ejecución de movimientos.

**Método**:
- Grabamos 5 videos adicionales con familiares del equipo
- Participantes con diferentes características físicas (altura, complexión, edad)
- Mismas 8 actividades del dataset original
- Protocolo de grabación estandarizado

**Ubicación**: `Entrega 2/videos familiaDiego/`
- 5 videos (vid1.mp4 a vid5.mp4)
- Anotaciones: `project-4-at-2025-11-02-12-56-089d6ad2.json`

### 2. Intercambio con Otro Grupo

**Objetivo**: Maximizar diversidad y obtener videos grabados bajo condiciones ligeramente diferentes.

**Método**:
- Colaboración con otro grupo del curso que tiene la misma consigna
- Intercambio de videos etiquetados
- Diferentes entornos de grabación (iluminación, fondo, ángulo de cámara)
- Diferentes estilos de ejecución de las actividades

**Ubicación**: `Entrega 2/videos otroGrupo/`
- 20 videos (vid1.mp4 a vid20.mp4)
- Anotaciones: `project-5-at-2025-11-02-14-50-27ad6e06.json`

### 3. Variabilidad Intencional en Posición de Brazos

**Problema Identificado**:
En los videos originales, las personas tendían a mantener los brazos en posiciones fijas durante "Standing", lo que podría llevar a los modelos a aprender que estar de pie está asociado con una posición específica de los brazos.

**Solución Implementada**:
En los nuevos videos (Entrega 2), se instruyó a los participantes para realizar **movimientos variados de brazos** durante las actividades estáticas:

- **Standing**: Brazos a los lados, brazos cruzados, manos en la cintura, brazos extendidos, etc.
- **Sitting**: Diferentes posiciones de manos (sobre las piernas, cruzadas, en el reposabrazos)

**Beneficio para el Modelo**:
- El modelo aprenderá que "estar de pie" no depende de la posición de los brazos
- Mayor robustez: el modelo se enfocará en características realmente discriminantes (posición de cadera, rodillas, ángulo del torso)
- Reducción de falsos positivos causados por posiciones de brazos específicas
- Mejor generalización a casos reales donde las personas tienen libertad de movimiento

## Proceso de Anotación

Todos los videos nuevos fueron anotados usando **Label Studio** con el mismo protocolo que Entrega 1:

1. Carga del video en Label Studio
2. Marcado de rangos temporales para cada actividad
3. Asignación de etiquetas: Standing, Sitting, Walk forward, Walk backward, Turn, Sit down, Get up
4. Exportación a JSON con estructura compatible

## Procesamiento y Combinación

El script `process_all_videos.py` se encarga de:

### Sincronización Automática

**Desafío**: Los archivos de video pueden estar desordenados en las carpetas, pero el JSON de Label Studio mantiene:
- `id`: Identificador único de cada video
- `file_upload`: Nombre original del archivo
- `annotations`: Rangos temporales con etiquetas

**Solución**: El script crea un mapeo automático entre:
- Los IDs del JSON ↔ Los nombres reales de los archivos en disco
- Los índices de frame de OpenCV ↔ Los índices de frame de Label Studio

Esto permite procesar correctamente cada video con sus anotaciones correspondientes, **independientemente del orden de los archivos**.

### Proceso por Fuente

1. **Entrega 1 (Videos APO)**:
   - Ruta: `Entrega 1/Videos APO/Videos APO/`
   - Labels: `Entrega 1/project-label-studio.json`
   - 18 videos originales
   - `source = "Entrega1"`

2. **Entrega 2 - Familia Diego**:
   - Ruta: `Entrega 2/videos familiaDiego/`
   - Labels: `project-4-at-2025-11-02-12-56-089d6ad2.json`
   - 5 videos nuevos
   - `source = "Entrega2_FamiliaDiego"`

3. **Entrega 2 - Otro Grupo**:
   - Ruta: `Entrega 2/videos otroGrupo/`
   - Labels: `project-5-at-2025-11-02-14-50-27ad6e06.json`
   - 20 videos compartidos
   - `source = "Entrega2_OtroGrupo"`

### Extracción de Features

Para cada frame de cada video:

1. **MediaPipe Pose** extrae 33 landmarks (x, y, z, visibility)
2. Se sincronizan los índices de frame (OpenCV vs Label Studio)
3. Se asigna la etiqueta según rangos temporales del JSON
4. Se calculan features derivadas:
   - Velocidades de articulaciones
   - Ángulos de rodillas y codos
   - Métricas de simetría
   - Características espaciales y temporales

### Combinación Final

Todos los datos se combinan en dos archivos CSV:

- `mediapipe_labels_dataset_combined.csv`: Dataset base con landmarks
- `mediapipe_labels_dataset_combined_enriched.csv`: Dataset con features derivadas adicionales

Cada registro incluye la columna `source` para rastrear el origen de los datos.

## Resultados del Dataset Combinado

### Volumen Total
- **Videos totales**: 43 (18 originales + 5 familia + 20 otro grupo)
- **Frames estimados**: ~30,000+ frames
- **Incremento**: ~3x respecto al dataset original

### Distribución por Fuente
- Entrega 1: ~40% de los datos
- Entrega 2 - Familia Diego: ~15% de los datos
- Entrega 2 - Otro Grupo: ~45% de los datos

### Diversidad Mejorada
- ✅ Mayor variedad de participantes (edad, género, altura, complexión)
- ✅ Diferentes entornos de grabación
- ✅ Variabilidad en la ejecución de movimientos
- ✅ **Posiciones de brazos variadas** durante actividades estáticas
- ✅ Diferentes estilos de cámara y ángulos

## Impacto Esperado en los Modelos

### Beneficios

1. **Mejor Generalización**:
   - Mayor diversidad → menor overfitting a características de sujetos específicos
   - Aprende patrones genuinos de movimiento en lugar de particularidades individuales

2. **Robustez a Variaciones de Brazos**:
   - El modelo aprenderá que "Standing" y "Sitting" no dependen de posición de brazos
   - Enfoque en características discriminantes reales (postura del torso, ángulos de piernas)

3. **Reducción de Sesgo**:
   - Videos de múltiples fuentes → menor sesgo de grabación
   - Diferentes condiciones de iluminación y fondo → mayor robustez

4. **Validación Más Realista**:
   - Leave-one-group-out: validar con datos de un grupo completo
   - Evaluar performance real en datos de sujetos nunca vistos

### Estrategia de Validación Sugerida

Para aprovechar las múltiples fuentes de datos:

```python
# Validación por fuente
for source in ['Entrega1', 'Entrega2_FamiliaDiego', 'Entrega2_OtroGrupo']:
    train_data = df[df['source'] != source]
    test_data = df[df['source'] == source]
    # Entrenar y evaluar...
```

Esto permite evaluar si el modelo generaliza a datos de grupos completamente nuevos.

## Conclusión

La estrategia de recolección de datos implementada en Entrega 2 amplía significativamente el dataset original mediante:

1. **Volumen**: Triplicando la cantidad de datos disponibles
2. **Diversidad**: Incorporando múltiples participantes y entornos
3. **Calidad**: Añadiendo variabilidad intencional en posiciones de brazos para mejorar la robustez del modelo
4. **Trazabilidad**: Manteniendo el origen de cada dato mediante la columna `source`

Esta mejora en los datos de entrenamiento debería resultar en modelos más robustos, generalizables y menos dependientes de características espurias como la posición de los brazos.
