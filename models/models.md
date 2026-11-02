 Realizados con 50k muestras y 2 epoch
 5.000 de validación.
5.000 de test.
Separación por película para evitar que las críticas de una misma película aparezcan en entrenamiento y test.

 # entrenamiento regresion
 - variables usadas:
 - review_content: texto de la crítica.
 - normalized_score: puntuación real de la crítica, normalizada entre 0 y 10.

 ## ¿Cuál es el problema?

El modelo tiene que aprender simultáneamente dos cosas:

- El sentimiento general: determinar si la crítica es positiva o negativa.
- La intensidad del sentimiento: determinar si corresponde a un 6, 7, 8, 9 o 10.

## resultados
| Métrica | Regresión directa |
| ------- | ----------------: |
| MAE     |             1,088 |
| RMSE    |             1,447 |

Conclusión: RoBERTa con regresión directa obtuvo peores resultados que un sistema que simplemente asignaba la media correspondiente a Fresh/Rotten.

# entrenamiento regresión residual
En lugar de pedirle a RoBERTa que prediga la puntuación completa, le proporcionamos una referencia inicial y hacemos que aprenda únicamente las desviaciones respecto a ella.

A partir del conjunto de entrenamiento calculamos las medias:
| Clasificación | Puntuación media |
| ------------- | ---------------: |
| Fresh         |            7,574 |
| Rotten        |            4,145 |

Estas medias representan la puntuación esperada de una crítica si únicamente conocemos su clasificación.

Por ejemplo, si una crítica es Fresh, nuestra predicción inicial sería 7,574.

Ahora, en lugar de entrenar el modelo para predecir la puntuación original, calculamos:

$$ \boxed{r=y_{\text{real}}-\mu_{\text{tipo}}} $$

Donde:

\(r\): residuo que queremos aprender.
\(y_{\text{real}}\): puntuación real de la crítica.
\(\mu_{\text{tipo}}\): media de las críticas Fresh o Rotten.

Ejemplo:

Tenemos una crítica Fresh con puntuación real de 9.

Su residuo sería:

r=9−7,574=+1,426

Por tanto, RoBERTa ya no tiene que aprender que el texto corresponde a una puntuación de 9.

Tiene que aprender que el texto expresa una valoración aproximadamente 1,426 puntos superior a la media de las críticas positivas.

para entrenar usamos:

- review_type: Fresh o Rotten.
- review_content: texto de la crítica.

Salida: Residuo respecto a la media de su clasificación.

## resultados
| Modelo                         |     MAE ↓ |    RMSE ↓ |
| ------------------------------ | --------: | --------: |
| Baseline Fresh/Rotten          |     1,044 |     1,324 |
| RoBERTa regresión directa      |     1,088 |     1,447 |
| **RoBERTa regresión residual** | **0,884** | **1,134** |

mejora:
Respecto al baseline:

Reducción del MAE: 15,3 %.
Reducción del RMSE: 14,3 %.

Respecto a la regresión directa:

Reducción del MAE: 18,7 %.
Reducción del RMSE: 21,6 %.

Es una mejora relevante, porque demuestra que el texto contiene información adicional a la clasificación binaria Fresh/Rotten.