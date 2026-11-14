# TC3009B Modulo 5 - Mario Alberto Pérez Barrera

# Clasificación de severidad de osteoartritis de rodilla (Kellgren-Lawrence) con Deep Learning

Proyecto de deep learning con Keras + TensorFlow que clasifica radiografías de rodilla según el grado de osteoartritis de la escala **Kellgren-Lawrence**, de 0 (normal) a 4 (severo).


---

## 1. Problema

La osteoartritis de rodilla afecta a millones de personas y su diagnóstico depende de que un radiólogo evalúe la radiografía. La escala K&L es el estándar para graduarla, pero la diferencia entre grados vecinos es sutil y subjetiva. Este proyecto entrena modelos de visión por computadora para **predecir el grado K&L a partir de la imagen**, como apoyo a un posible sistema de diagnóstico asistido.

Es un problema de clasificación multiclase (5 clases):

| Grado | Clase |
|---|---|
| 0 | Normal |
| 1 | Dudoso |
| 2 | Leve |
| 3 | Moderado |
| 4 | Severo |

## 2. Dataset

**Digital Knee X-ray Images** (https://www.kaggle.com/datasets/orvile/digital-knee-x-ray-images/data): 1,650 radiografías digitales de rodilla en escala de grises, anotadas por dos expertos médicos.

- Se detectaron y eliminaron **17 imágenes duplicadas**, quedando **1,633 imágenes únicas**, para evitar fuga de datos entre entrenamiento y prueba.
- Distribución de clases (desbalanceada):

| 0 Normal | 1 Dudoso | 2 Leve | 3 Moderado | 4 Severo |
|---|---|---|---|---|
| 513 | 474 | 227 | 213 | 206 |

- **Split estratificado:** 70% entrenamiento (1,143), 15% validación (245), 15% prueba (245).
- **Preprocesamiento:** conversión a escala de grises, redimensionado a 224×224 y normalización dentro del modelo.

## 3. Estructura del repositorio

```
.
├── knee_xray_kl_grading.ipynb 
├── requirements.txt
└── README.md
```

**El dataset se descarga desde Kagglehub**

## 4. Cómo ejecutarlo
1. Instalar dependencias:
   bash
   pip install -r requirements.txt
2. Ejecuta `knee_xray_kl_grading.ipynb` utilizando Collab o un entorno con GPU.
3. Parámetros editables en la primera celda: `EXPERT` (`"I"`, `"II"` o `"consensus"`), `GROUP` (5 clases o agrupación en 3) y `IMG`/`BATCH`.

## 5. Arquitectura

- **CNN desde cero:** bloques `Conv2D → BatchNorm → ReLU → MaxPooling`, `GlobalAveragePooling` y capas densas con salida `softmax` de 5 clases.
- **MobileNetV2 (transfer learning):** base preentrenada en ImageNet, con la imagen en escala de grises replicada a 3 canales, seguida de `GlobalAveragePooling`, `Dropout` y capa `softmax`.

Detalles de entrenamiento: optimizador Adam, pérdida `sparse_categorical_crossentropy`, **class weights** para compensar el desbalance, `EarlyStopping` y `ReduceLROnPlateau` sobre la pérdida de validación. El mejor modelo se elige por **macro-F1 de validación**.

## 6. Experimentos y resultados

| # | Experimento | Acc test | F1 macro test | QWK test |
|---|---|---|---|---|
| 1 | CNN base (sin regularización ni aumento de datos) | 0.314 | 0.156 | 0.053 |
| 2 | CNN + Dropout + L2 + aumento de datos | 0.261 | 0.158 | 0.174 |
| 3 | Búsqueda de lr × dropout sobre la CNN (4 combinaciones) | 0.127 – 0.290 | 0.045 – 0.090 | 0.000 – 0.022 |
| 4 | MobileNetV2 con base congelada + aumento de datos | 0.592 | 0.591 | 0.739 |
| 5 | **MobileNetV2 con fine-tuning (últimas ~30 capas, lr = 1e-5)** | **0.629** | **0.626** | **0.801** |


**QWK** (Kappa cuadrático ponderado) penaliza más los errores entre grados lejanos, por lo que es una métrica adecuada para una escala ordinal como K&L.

### Mejor modelo (MobileNetV2 + fine-tuning)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| 0 Normal | 0.60 | 0.86 | 0.71 |
| 1 Dudoso | 0.66 | 0.32 | 0.43 |
| 2 Leve | 0.51 | 0.53 | 0.52 |
| 3 Moderado | 0.69 | 0.69 | 0.69 |
| 4 Severo | 0.76 | 0.81 | 0.78 |



## 7. Uso del modelo para predecir

Al final del notebook, la función `predict_image(ruta)` muestra la probabilidad de cada grado para una radiografía nueva:

```python
predict_image("ruta/a/radiografia.png")
```

## 11. Créditos

- Dataset: *Digital Knee X-ray Images*, autor `orvile` en Kaggle (CC BY 4.0).
- Modelo base preentrenado: MobileNetV2 (Keras Applications).
