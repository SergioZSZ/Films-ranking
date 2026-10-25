# Films-ranking


## 2. Datasets
## 2.1 sIDMB dataset




## 2.2 Rotten Tomatoes Dataset
son 2 csv, hace falta 

### 2.2.1 rotten_tomatoes_movies.csv
películas del dataset
hace falta:
- normalizar las scores en una escala 0-10 de cada reseña (nuevo campo)
- calcular la puntuacion media de cada pelicula
- realizar regulacion bayesiana de la score respecto a valoraciones/puntuacion (nuevo campo)
- rellenar los campos vacios con analisis de sentimientos
![alt text](images/regulacion_bayesiana.png)


| Nº | Key                                | Tipo    | Descripción                                                                  |
| -: | ---------------------------------- | ------- | ---------------------------------------------------------------------------- |
|  1 | `rotten_tomatoes_link`             | String  | Identificador único de la película en Rotten Tomatoes. Ejemplo: `m/0814255`. |
|  2 | `movie_title`                      | String  | Título de la película.                                                       |
|  3 | `movie_info`                       | String  | Sinopsis o descripción argumental de la película.                            |
|  4 | `critics_consensus`                | String  | Resumen del consenso general de la crítica profesional sobre la película.    |
|  5 | `content_rating`                   | String  | Clasificación por edades: G, PG, PG-13, R, NC-17, NR.                        |
|  6 | `genres`                           | String  | Géneros cinematográficos separados por comas.                                |
|  7 | `directors`                        | String  | Director o directores de la película.                                        |
|  8 | `authors`                          | String  | Guionistas o autores de la película.                                         |
|  9 | `actors`                           | String  | Actores participantes.                                                       |
| 10 | `original_release_date`            | Date    | Fecha de estreno original registrada en Rotten Tomatoes.                     |
| 11 | `streaming_release_date`           | Date    | Fecha de lanzamiento en plataformas digitales o streaming.                   |
| 12 | `runtime`                          | Float   | Duración de la película en minutos.                                          |
| 13 | `production_company`               | String  | Productora o distribuidora cinematográfica.                                  |
| 14 | `tomatometer_status`               | String  | Clasificación profesional: `Rotten`, `Fresh` o `Certified-Fresh`.            |
| 15 | `tomatometer_rating`               | Float   | Porcentaje de críticas profesionales positivas (0–100).                      |
| 16 | `tomatometer_count`                | Float   | Número de críticas profesionales contabilizadas en el Tomatometer.           |
| 17 | `audience_status`                  | String  | Clasificación de la recepción del público: `Upright` o `Spilled`.            |
| 18 | `audience_rating`                  | Float   | Porcentaje de valoraciones positivas del público (0–100).                    |
| 19 | `audience_count`                   | Float   | Número de valoraciones del público.                                          |
| 20 | `tomatometer_top_critics_count`    | Integer | Número de críticas realizadas por críticos destacados (*Top Critics*).       |
| 21 | `tomatometer_fresh_critics_count`  | Integer | Número de críticas profesionales clasificadas como positivas.                |
| 22 | `tomatometer_rotten_critics_count` | Integer | Número de críticas profesionales clasificadas como negativas.                |


## 2.2.2 rotten_tomatoes_critic_reviews.csv
 criticas de las peliculas del anterior dataset

 | Nº | Key                    | Tipo    | Descripción                                                                                         |
| -: | ---------------------- | ------- | --------------------------------------------------------------------------------------------------- |
|  1 | `rotten_tomatoes_link` | String  | Identificador de la película a la que pertenece la crítica. Permite relacionarla con `movies.csv`.  |
|  2 | `critic_name`          | String  | Nombre del crítico profesional que realizó la reseña.                                               |
|  3 | `top_critic`           | Boolean | Indica si el crítico está considerado *Top Critic* por Rotten Tomatoes.                             |
|  4 | `publisher_name`       | String  | Medio de comunicación donde se publicó la crítica.                                                  |
|  5 | `review_type`          | String  | Clasificación de la crítica: `Fresh` (positiva) o `Rotten` (negativa).                              |
|  6 | `review_score`         | String  | Puntuación original del crítico. Puede tener formatos como `4/5`, `7/10`, `B+`, `A-` o estar vacía. |
|  7 | `review_date`          | Date    | Fecha de publicación de la crítica.                                                                 |
|  8 | `review_content`       | String  | Texto de la reseña profesional. Será la entrada del modelo NLP.                                     |

## 2.2.3 columnas que necesitaremos para fusionarlos
| Dataset | Key                     | Función en EPAC                                                |
| ------- | ----------------------- | -------------------------------------------------------------- |
| Movies  | `rotten_tomatoes_link`  | Identificar películas y relacionar ambos CSV.                  |
| Movies  | `movie_title`           | Relacionar películas con IMDb.                                 |
| Movies  | `original_release_date` | Desambiguar películas con títulos iguales.                     |
| Movies  | `tomatometer_rating`    | Comparar nuestra puntuación con la oficial de Rotten Tomatoes. |
| Movies  | `tomatometer_count`     | Conocer el número de críticas profesionales.                   |
| Movies  | `directors` | Ayuda a resolver correspondencias ambiguas con IMDb. |
| Movies  | `tomatometer_rating` |  Comparar nuestra puntuación con Rotten Tomatoes.| 
| Reviews | `rotten_tomatoes_link`  | Asociar cada crítica a su película.                            |
| Reviews | `review_score`          | Obtener las puntuaciones numéricas originales.                 |
| Reviews | `review_type`           | Conocer si la crítica es positiva o negativa.                  |
| Reviews | `review_content`        | Aplicar RoBERTa a las críticas sin puntuación.                 |
| Reviews | `top_critic`            | Explorar si interesa diferenciar a los críticos destacados.    |
