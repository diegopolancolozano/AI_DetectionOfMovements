# Universidad ICESI  

## Facultad de Ingeniería, Diseño y Ciencias Aplicadas  
**Departamento de Computación y Sistemas Inteligentes**  
**Programa:** Ingeniería de Sistemas  
**Asignatura:** APO3  
**Semestre:** 2025-2  

---

## Lineamientos para el Proyecto Final – Inteligencia Artificial I  

El proyecto final del curso **Inteligencia Artificial I** es un **trabajo grupal** (mínimo 2 y máximo 3 estudiantes por grupo) cuyo objetivo es desarrollar una **solución a un problema real** utilizando modelos de analítica y conjuntos de datos en diferentes formatos.

Cada grupo debe:  
- Comprender el problema y su contexto.  
- Investigar antecedentes.  
- Definir una metodología de trabajo.  
- Proponer métricas de desempeño.  
- Entrenar y evaluar diferentes modelos de analítica.  
- Ajustar hiperparámetros y evaluar los resultados con métricas predefinidas.  

Se debe usar la metodología **CRISP-DM**, adaptada a las necesidades del proyecto.

---

## 1. Caso de Estudio Propuesto: Sistema de Anotación de Video  

### Objetivo
Desarrollar una herramienta de software capaz de analizar **actividades específicas de una persona** (caminar hacia la cámara, regresar, girar, sentarse, ponerse de pie) y realizar un **seguimiento de movimientos articulares y posturales**.

### Requerimientos Técnicos
- **Entradas:** Video en tiempo real capturado por la cámara.  
- **Salidas:** Clasificación de la actividad en tiempo real y análisis de inclinaciones laterales y movimientos articulares (muñecas, rodillas, caderas).  

---

### Recolección de Datos y Anotación  

**Base de datos:**  
Captura de videos con varias personas realizando las actividades desde diferentes perspectivas y velocidades.  

**Anotación:**  
- **Manual:** Etiquetar segmentos donde ocurren las actividades clave.  
- **Automática:** Usar herramientas como [LabelStudio](https://labelstud.io/) o [CVAT](https://cvat.ai) para asignar etiquetas a secuencias de video.  

---

### Seguimiento de Articulaciones y Movimientos  

- **Herramientas recomendadas:**  
  - [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)  
  - [OpenPose](https://quickpose.ai/faqs/mediapipe-vs-openpose/)  
- **Landmarks a seguir:** Cadera, rodillas, tobillos, muñecas, hombros, cabeza.  
- **Inclinación lateral:** Comparar posiciones de hombros y caderas para medir inclinaciones del tronco.  
- **Movimientos:** Calcular ángulos de flexión/extensión en rodillas y caderas a partir de los cambios en posiciones articulares.  

---

### Preprocesamiento de Datos  

- **Normalización:** Estandarizar coordenadas de articulaciones (independiente de altura o distancia).  
- **Filtrado:** Aplicar filtros suaves para eliminar ruido.  
- **Generación de características:**  
  - Velocidad de las articulaciones.  
  - Ángulos relativos entre articulaciones.  
  - Inclinación del tronco.  

---

### Entrenamiento del Sistema de Clasificación  

- **Modelos supervisados recomendados:** SVM, Random Forest, XGBoost.  
- **Entrenamiento:**  
  - Dividir datos en conjuntos de entrenamiento y prueba.  
  - Clasificar actividades usando posiciones, velocidades, ángulos, etc.  

---

### Inferencia en Tiempo Real  

Implementar una **visualización en tiempo real** de la actividad detectada y medidas posturales (por ejemplo, inclinaciones o ángulos articulares).  

---

### Entregable  

Desarrollar una **interfaz gráfica sencilla** que permita al usuario ver en tiempo real:  
- La actividad detectada.  
- Los ángulos de las articulaciones.  

---

### Validación y Evaluación  

- Probar con diferentes personas.  
- Comparar predicciones del sistema con etiquetas reales.  
- Calcular métricas: **Precisión, Recall y F1-Score.**  

---

### Recursos Clave  

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)  
- [LabelStudio](https://labelstud.io/)  
- [Comparativa CVAT vs LabelStudio](https://medium.com/cvat-ai/cvat-vs-labelstudio-which-one-is-better-b1a0d333842e)  

---

## 2. Evaluación y Entregables  

La calidad del trabajo se evaluará respondiendo preguntas como:  
- ¿La metodología es clara y robusta?  
- ¿Son razonables las aproximaciones realizadas?  
- ¿Se exploraron y procesaron adecuadamente los datos?  
- ¿Las soluciones propuestas son ingeniosas e interesantes?  
- ¿Explican correctamente los impactos del proyecto?  
- ¿Se complementaron los datos iniciales?  
- ¿Se evidencian conocimientos no triviales sobre el problema y los modelos?  
- ¿El trabaj
