# TC3009B Modulo 5 - Mario Alberto Pérez Barrera

# Clasificación de severidad de osteoartritis de rodilla (Kellgren-Lawrence) con Deep Learning

Proyecto de deep learning con **Keras + TensorFlow** que clasifica radiografías de rodilla según el grado de osteoartritis de la escala **Kellgren-Lawrence (K&L)**, de 0 (normal) a 4 (severo).

> **Autor(es):** [COMPLETAR]
> **Curso / Módulo:** [COMPLETAR]
> **Notebook principal:** `knee_xray_kl_grading.ipynb`

---

## 1. Problema

La osteoartritis de rodilla afecta a millones de personas y su diagnóstico depende de que un radiólogo evalúe la radiografía. La escala K&L es el estándar para graduarla, pero la diferencia entre grados vecinos es sutil y subjetiva. Este proyecto entrena modelos de visión por computadora para **predecir el grado K&L a partir de la imagen**, como apoyo a un posible sistema de diagnóstico asistido.

Es un problema de **clasificación multiclase (5 clases)**:

| Grado | Clase |
|---|---|
| 0 | Normal |
| 1 | Dudoso |
| 2 | Leve |
| 3 | Moderado |
| 4 | Severo |

## 2. Dataset

**Digital Knee X-ray Images** (Kaggle, `orvile/digital-knee-x-ray-images`, licencia CC BY 4.0): 1,650 radiografías digitales de rodilla en escala de grises, anotadas por dos expertos médicos (carpetas `MedicalExpert-I` y `MedicalExpert-II`).

- Se usó la anotación del **experto I**.
- Se detectaron y eliminaron **17 imágenes duplicadas** (por hash del archivo), quedando **1,633 imágenes únicas**, para evitar fuga de datos entre entrenamiento y prueba.
- Distribución de clases (desbalanceada):

| 0 Normal | 1 Dudoso | 2 Leve | 3 Moderado | 4 Severo |
|---|---|---|---|---|
| 513 | 474 | 227 | 213 | 206 |

- **Split estratificado:** 70% entrenamiento (1,143), 15% validación (245), 15% prueba (245).
- **Preprocesamiento:** conversión a escala de grises, redimensionado a 224×224 y normalización dentro del modelo.

## 3. Estructura del repositorio

```
.
├── knee_xray_kl_grading.ipynb   # Notebook con todo el flujo (datos, modelos, análisis, predicciones)
├── requirements.txt
└── README.md
```

El dataset **no** se incluye en el repositorio; el notebook lo descarga con `kagglehub`.

## 4. Cómo ejecutarlo

1. Clona el repositorio:
   ```bash
   git clone [URL DEL REPO]
   cd [NOMBRE DEL REPO]
   ```
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abre `knee_xray_kl_grading.ipynb` y ejecútalo de arriba hacia abajo. Se recomienda **GPU** (Google Colab o Kaggle) e **internet activado**, ya que el notebook descarga el dataset y los pesos preentrenados de MobileNetV2.
4. Parámetros editables en la primera celda: `EXPERT` (`"I"`, `"II"` o `"consensus"`), `GROUP` (5 clases o agrupación en 3) y `IMG`/`BATCH`.

## 5. Arquitecturas

- **CNN desde cero:** bloques `Conv2D → BatchNorm → ReLU → MaxPooling`, `GlobalAveragePooling` y capas densas con salida `softmax` de 5 clases.
- **MobileNetV2 (transfer learning):** base preentrenada en ImageNet, con la imagen en escala de grises replicada a 3 canales, seguida de `GlobalAveragePooling`, `Dropout` y capa `softmax`.

Detalles de entrenamiento: optimizador Adam, pérdida `sparse_categorical_crossentropy`, **class weights** para compensar el desbalance, `EarlyStopping` y `ReduceLROnPlateau` sobre la pérdida de validación. El mejor modelo se elige por **macro-F1 de validación**, nunca por prueba.

## 6. Experimentos y resultados

| # | Experimento | Acc test | F1 macro test | QWK test |
|---|---|---|---|---|
| 1 | CNN base (sin regularización ni aumento de datos) | 0.314 | 0.156 | 0.053 |
| 2 | CNN + Dropout + L2 + aumento de datos | 0.261 | 0.158 | 0.174 |
| 3 | Búsqueda de lr × dropout sobre la CNN (4 combinaciones) | 0.127 – 0.290 | 0.045 – 0.090 | 0.000 – 0.022 |
| 4 | MobileNetV2 con base congelada + aumento de datos | 0.592 | 0.591 | 0.739 |
| 5 | **MobileNetV2 con fine-tuning (últimas ~30 capas, lr = 1e-5)** | **0.629** | **0.626** | **0.801** |
| [COMPLETAR] | [Resultados de la CNN v2 u otros experimentos adicionales] | | | |

**QWK** (Kappa cuadrático ponderado) penaliza más los errores entre grados lejanos, por lo que es una métrica adecuada para una escala ordinal como K&L.

### Mejor modelo (MobileNetV2 + fine-tuning)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| 0 Normal | 0.60 | 0.86 | 0.71 |
| 1 Dudoso | 0.66 | 0.32 | 0.43 |
| 2 Leve | 0.51 | 0.53 | 0.52 |
| 3 Moderado | 0.69 | 0.69 | 0.69 |
| 4 Severo | 0.76 | 0.81 | 0.78 |

Matriz de confusión, curvas de entrenamiento y ejemplos de predicción están en el notebook. [COMPLETAR: agregar imágenes a una carpeta `images/` y enlazarlas aquí]

## 7. Análisis: qué funcionó y por qué

- **Transfer learning fue el cambio que más mejoró el desempeño.** Con ~1,100 imágenes de entrenamiento, los *features* generales de ImageNet (bordes, texturas) aportan mucho más que aprender desde cero. Las CNN entrenadas desde cero no lograron aprender en las configuraciones probadas.
- **El fine-tuning aportó una mejora moderada** (F1 de 0.59 a 0.63; QWK de 0.74 a 0.80) al adaptar las capas finales a las características de las radiografías sin destruir lo preentrenado. Con solo 245 imágenes de prueba, esta diferencia debe interpretarse con cautela.
- **La regularización y el aumento de datos sobre la CNN desde cero no funcionaron**, probablemente por usar demasiada regularización a la vez sobre una red pequeña con pocos datos (subajuste). Por eso la búsqueda de hiperparámetros del experimento 3 no es concluyente. [COMPLETAR si se reentrenó la CNN con otra configuración]
- **Los errores se concentran en grados vecinos**, como indica el QWK de 0.80. La clase más difícil es **1 Dudoso**, que se confunde con Normal y con Leve; esto es consistente con que la frontera entre los grados 0 y 1 es la más subjetiva incluso para especialistas.
- **Manejo del desbalance:** class weights y métricas por clase (macro-F1, matriz de confusión) en lugar de accuracy únicamente.

## 8. Limitaciones

- El dataset **no incluye identificador de paciente**, por lo que no se puede garantizar que radiografías de un mismo paciente no estén repartidas entre entrenamiento y prueba.
- Solo 245 imágenes de prueba: los resultados tienen un margen de error considerable.
- Dataset de una sola fuente (mismo equipo de rayos X); la generalización a otros hospitales es incierta.
- Las etiquetas del experto I y II pueden discrepar; se usó únicamente una anotación.
- Este proyecto es **académico**: no es una herramienta clínica ni sustituye el criterio de un profesional de la salud.

## 9. Trabajo futuro

- Reentrenar la CNN desde cero con menos regularización y comparar contra transfer learning.
- Ajustar hiperparámetros del modelo ganador (lr de fine-tuning, número de capas descongeladas).
- Probar otras arquitecturas (EfficientNet, DenseNet121) y ensambles.
- Agrupar a 3 clases (sano / leve / severo) con el parámetro `GROUP`.
- Tratar el problema como **regresión u ordinal**, aprovechando el orden de los grados.
- Incorporar la corrección recibida en la entrega intermedia. [COMPLETAR]

## 10. Uso del modelo para predecir

Al final del notebook, la función `predict_image(ruta)` muestra la probabilidad de cada grado para una radiografía nueva:

```python
predict_image("ruta/a/radiografia.png")
```

## 11. Créditos

- Dataset: *Digital Knee X-ray Images*, autor `orvile` en Kaggle (CC BY 4.0).
- Modelo base preentrenado: MobileNetV2 (Keras Applications).
