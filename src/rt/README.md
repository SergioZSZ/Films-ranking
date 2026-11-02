# Rotten Tomatoes — Normalización, RoBERTa residual e inferencia (0-10)

## Objetivo

Entrenar un modelo `FacebookAI/roberta-base` para estimar la diferencia entre la
nota normalizada de una crítica y la media de su clase Fresh/Rotten. No se usan
probabilidades de clasificación como puntuaciones. En inferencia, el residual
predicho se suma a la media de la clase y el resultado se recorta a `[0, 10]`.

## Preparación

Trabajar desde Python 3.12 en un entorno virtual. Instalar PyTorch según la plataforma/GPU siguiendo la guía oficial de PyTorch; después:

```bash
pip install pandas numpy scikit-learn datasets transformers accelerate
```

Si no se dispone de GPU CUDA, reducir `MAX_TRAIN` a 500 para una prueba técnica, pues el fine-tuning completo en CPU puede resultar muy lento.

Organización relevante:

```text
Films-ranking/
├── processed_datasets/rottentomatoes/
│   └── rotten_tomatoes_critic_reviews_clean.csv
├── models/
│   └── epac-roberta-residual-full/
└── src/rt/
    ├── rt_processment.py
    ├── rt_roberta_train_full.py
    └── predict_missing_scores.py
```

Las rutas de inferencia se pueden sustituir con `--input`, `--model` y
`--output` sin editar el script.

## Ejecución recomendada

1. Generar el CSV limpio con `normalized_score` mediante el preprocesamiento.
2. Entrenar `src/rt/rt_roberta_train_full.py` en Kaggle.
3. Descargar la carpeta completa en `models/epac-roberta-residual-full/`.
4. Validar el artefacto y los datos antes de cargar los pesos.
5. Ejecutar una prueba limitada y, después, la inferencia completa.

## Artefacto que debe volver de Kaggle

La carpeta del modelo debe contener al menos:

```text
models/epac-roberta-residual-full/
├── config.json
├── model.safetensors                 # o pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json                    # o vocab.json + merges.txt
└── training_metadata.json
```

`training_metadata.json` debe tener `status: completed`, las medias `Fresh` y
`Rotten`, `max_length` y el contrato exacto de objetivo, template y
postprocesado generado por el entrenamiento full. Las medias no se duplican en
el script de inferencia: se leen siempre de este archivo.

Validación sin cargar los pesos ni escribir salida:

```bash
poetry run python src/rt/predict_missing_scores.py --validate-only
```

Prueba corta con las primeras 100 reseñas elegibles:

```bash
poetry run python src/rt/predict_missing_scores.py --limit 100
```

Inferencia completa:

```bash
poetry run python src/rt/predict_missing_scores.py
```

Por defecto genera
`processed_datasets/rottentomatoes/rotten_tomatoes_critic_reviews_scored.csv`.
Se puede forzar CPU o CUDA con `--device cpu` o `--device cuda`, y ajustar los
lotes con `--batch-size`.

El CSV conserva todas las columnas originales y añade:

- `predicted_score`: nota reconstruida, solo para filas inferidas;
- `final_score`: nota normalizada original o, si no existe, la predicción;
- `score_source`: procedencia exacta de cada fila;
- `prediction_model` y `prediction_metadata`: trazabilidad del artefacto.

Las filas elegibles no procesadas por `--limit` quedan como
`pending_prediction_limit`, para que una prueba corta no parezca una inferencia
completa.

## Cómo hemos diseñado el experimento

- Entrenar únicamente con `review_score` en forma explícita `3/4`, `4/5`, `7/10`, etc. No alimentar con etiquetas de letra transformadas mediante una escala subjetiva en el primer experimento.
- Construir la entrada exacta `Sentiment classification: {review_type}. Review: {review_content}`. `review_type` también selecciona la media que se suma al residual predicho.
- Separar train/valid/test **por película** (con la clave `rotten_tomatoes_link`) antes de muestrear. También se excluyen textos idénticos de train si coinciden con textos en valid/test.
- Métricas: MAE y RMSE, en la escala 0-10. Reportar resultados del test una sola vez por experimento definido.
- Entrenar inicialmente con 50 000 reseñas y escalar después si el experimento es prometedor.
- Las predicciones se recortan a `[0, 10]` y solo se generan para filas con `review_score` y `normalized_score` vacíos, texto no vacío y tipo Fresh/Rotten válido.
- Las notas existentes normalizadas no se sobrescriben. `final_score` elige `normalized_score` o `predicted_score`; `score_source` registra su procedencia.

## Limitaciones científicas importantes

- El score de un clasificador binario es confianza de sentimiento, no intensidad de calidad de película; por eso entrenamos regresión.
- Las notas alfabéticas del preprocesamiento son aproximaciones **no oficiales**: se deben validar y realizar análisis de sensibilidad para el ranking. Los números aislados (por ejemplo `4`, `75`) se mantienen sin normalizar, porque su escala es ambigua.
- Las reseñas sin puntuación pueden diferir sistemáticamente de las que tienen nota (sesgo de ausencia de datos). Comparar las distribuciones por fecha, crítico, medio y `review_type` antes de generalizar el modelo.
- Un MAE bajo no garantiza mejoras en el ranking; hay que evaluar por separado su efecto sobre la fórmula final y sus posibles sesgos.
- Registrar versión de bibliotecas, modelo, fuente de datos y semilla en los resultados definitivos para reproducibilidad.

## Fuentes

- Modelo base: https://huggingface.co/FacebookAI/roberta-base
- Robertson y Leone (2024), *A Dataset for Multi-Scale Film Rating Inference from Reviews*: https://aclanthology.org/2024.nlperspectives-1.16/
- Guía Trainer de Hugging Face: https://huggingface.co/docs/transformers/main_classes/trainer



## datos de ldataset que roberta puede sacar

273.605 no tienen review_score.

    273.215 no tienen score pero sí texto: RoBERTa puede predecirlas.
    390 no tienen score ni texto: RoBERTa no puede inferirlas.
    1.459 tienen un score original no vacío que el normalizador actual no reconoce: no se usan para entrenar ni predecir y quedan como unavailable_unparseable.
