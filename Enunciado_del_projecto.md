# Proyecto Final - Algoritmos y Programación III (APO3)

**Universidad ICESI**
Facultad de Ingeniería, Diseño y Ciencias Aplicadas
Departamento de Computación y Sistemas Inteligentes
**Asignatura:** APO3
**Semestre:** 2025-2

---

## Lineamientos del Proyecto Final

El proyecto final es un trabajo grupal (mínimo 2 y máximo 3 estudiantes) que busca desarrollar una solución a un problema real usando modelos de analítica y conjuntos de datos en diferentes formatos.

Cada grupo debe:

* Entender el problema y su contexto.
* Investigar antecedentes y definir metodología.
* Proponer métricas de desempeño para evaluar el progreso.
* Entrenar y evaluar modelos de analítica ajustando hiperparámetros y midiendo resultados.
* Utilizar la metodología **CRISP-DM**, adaptada a su proyecto.

---

## 1. Caso de Estudio: Sistema de Anotación de Video

### Objetivo

Desarrollar una herramienta de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, girar, sentarse, ponerse de pie) y realizar un seguimiento de movimientos articulares y posturales.

### Requerimientos Técnicos

* **Entradas:** Video en tiempo real capturado por la cámara.
* **Salidas:** Clasificación de la actividad en tiempo real y análisis de inclinaciones laterales y movimientos de articulaciones clave (muñecas, rodillas, caderas).

---

### Recolección de Datos y Anotación

**Base de datos:**
Captura de videos con varias personas realizando las actividades desde diferentes perspectivas y velocidades.

**Anotación:**

* **Manual:** Etiquetar segmentos donde ocurren las actividades clave.
* **Automática:** Usar herramientas como **LabelStudio** o **CVAT** para facilitar la anotación de eventos.

---

### Seguimiento de Articulaciones y Movimientos

* **Herramientas:** [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419) o [OpenPose](https://quickpose.ai/faqs/mediapipe-vs-openpose/)
* **Landmarks a seguir:** Cadera, rodillas, tobillos, muñecas, hombros, cabeza.
* **Inclinación lateral:** Comparar la posición de hombros y caderas.
* **Movimientos:** Calcular ángulos articulares (flexión/extensión) mediante flujo de posiciones.

---

### Preprocesamiento de Datos

* **Normalización:** Estandarizar coordenadas para eliminar dependencia de altura o distancia de cámara.
* **Filtrado:** Aplicar filtros suaves para reducir ruido.
* **Extracción de características:**

  * Velocidad de articulaciones.
  * Ángulos relativos.
  * Inclinación del tronco.

---

### Entrenamiento del Sistema de Clasificación

* **Modelos supervisados:** SVM, Random Forest, XGBoost, entre otros.
* **Entrenamiento:** Dividir los datos en entrenamiento y prueba.
* **Características:** Posiciones, velocidades, ángulos, etc.

---

### Inferencia en Tiempo Real

Mostrar visualmente la actividad detectada y medidas posturales (por ejemplo, inclinación del tronco).

**Entregable:**
Una interfaz gráfica sencilla para visualizar la actividad y los ángulos articulares en tiempo real.

---

### Validación y Evaluación

* Pruebas con varias personas y comparación con etiquetas reales.
* Métricas: precisión, recall, F1-Score.

**Recursos clave:**

* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)
* [LabelStudio](https://labelstud.io/)
* [Comparación CVAT vs LabelStudio](https://medium.com/cvat-ai/cvat-vs-labelstudio-which-one-is-better-b1a0d333842e)

---

## 2. Evaluación y Entregables

Se evaluará con base en:

* Claridad y robustez metodológica.
* Pertinencia de las aproximaciones.
* Calidad del procesamiento de datos.
* Ingenio e interés de la solución.
* Capacidad de análisis y discusión de resultados.
* Desarrollo de competencias del curso.

### Outcome 6

**Habilidad para desarrollar y conducir experimentos, analizar e interpretar datos, y usar juicio ingenieril para obtener conclusiones.**

---

## Cronograma de Entregas

### Entrega 1 – Semana 12

* Preguntas de interés y tipo de problema.
* Metodología y métricas.
* Datos recolectados y análisis exploratorio.
* Estrategias para aumentar datos.
* Análisis ético del uso de IA en el contexto del problema.

### Entrega 2 – Semana 14

* Estrategia de obtención y preparación de datos.
* Entrenamiento y ajuste de modelos.
* Resultados y plan de despliegue.
* Análisis inicial de impactos de la solución.

### Entrega 3 – Semana 17

* Reducción de características.
* Evaluación final y despliegue.
* Presentación al “cliente”.
* Video de máximo 10 minutos con contexto, técnicas, resultados y logros.
* Análisis final de impactos.

---

## Aspectos Clave

1. **Código documentado:** Si se usa código o datos de terceros, referenciarlos claramente.
2. **Informes claros y concisos:** Incluir diagramas de flujo, arquitectura y resultados con gráficos vectoriales.

---

## Estructura del Reporte Final (máx. 7 páginas)

1. **Título**
2. **Resumen (Abstract)**
3. **Introducción:** Contexto, descripción del problema y su relevancia.
4. **Marco Teórico:** Conceptos necesarios para entender el desarrollo.
5. **Metodología:** Descripción del enfoque (sin repetir CRISP-DM textual).
6. **Resultados:** Rendimiento y métricas obtenidas.
7. **Análisis de Resultados:** Discusión, sobreajuste, comparaciones con la literatura.
8. **Conclusiones y Trabajo Futuro.**
9. **Referencias:** Formato IEEE.

**Recomendación:** Revisar artículos de conferencias como NIPS, ICML o ICLR antes de redactar el informe final.

