# Films Ranking

## 1. Introducción

Este proyecto construye un ranking histórico de películas combinando dos puntos
de vista complementarios:

- **IMDb** representa la valoración del público general mediante una nota y su
  número de votos.
- **Rotten Tomatoes** aporta críticas profesionales, tanto su texto como la
  puntuación asignada por cada crítico.

El flujo prepara ambos datasets, completa con RoBERTa las notas profesionales
ausentes, relaciona las películas entre las dos fuentes y almacena el resultado
en PostgreSQL.

La fórmula final parte de la nota bayesiana de IMDb y la ajusta con la media de
los críticos: así combina público y crítica sin dar demasiado peso a unas pocas
reseñas. Su cálculo se explica en el apartado 3.3.

## 2. Datasets

### 2.1 IMDb

Se utilizan los [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/),
distribuidos como archivos TSV comprimidos. IMDb aporta títulos, años,
directores, puntuaciones de usuarios y número de votos.

| Archivo | Variables utilizadas | Uso |
|---|---|---|
| `title.basics.tsv.gz` | `tconst`, `titleType`, `primaryTitle`, `originalTitle`, `startYear` | Identificar películas y sus títulos. |
| `title.ratings.tsv.gz` | `tconst`, `averageRating`, `numVotes` | Obtener la nota de usuarios y su volumen de votos. |
| `title.crew.tsv.gz` | `tconst`, `directors` | Asociar cada película con sus directores. |
| `name.basics.tsv.gz` | `nconst`, `primaryName` | Convertir los identificadores de director en nombres. |

`averageRating` refleja votos de usuarios de IMDb, no crítica profesional.

### 2.2 Rotten Tomatoes

Se utiliza el dataset
[Rotten Tomatoes Movies and Critic Reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset),
formado por dos CSV relacionados mediante `rotten_tomatoes_link`.

#### Películas: `rotten_tomatoes_movies.csv`

| Variable | Descripción |
|---|---|
| `rotten_tomatoes_link` | Identificador de la película. |
| `movie_title` | Título. |
| `original_release_date` | Fecha de estreno, útil para desambiguar. |
| `directors` | Directores, utilizados también en el emparejamiento con IMDb. |
| `tomatometer_rating` | Porcentaje de críticas clasificadas como positivas. |
| `tomatometer_count` | Número de críticas incluidas en el Tomatometer. |

El Tomatometer mide el porcentaje de críticas positivas; no es la media de las
notas numéricas de los críticos.

#### Críticas: `rotten_tomatoes_critic_reviews.csv`

| Variable | Descripción |
|---|---|
| `rotten_tomatoes_link` | Película a la que pertenece la crítica. |
| `critic_name` | Nombre del crítico. |
| `publisher_name` | Medio de publicación. |
| `top_critic` | Indica si pertenece al grupo *Top Critics*. |
| `review_type` | Clasificación `Fresh` o `Rotten`. |
| `review_score` | Nota original, que puede usar fracciones, porcentajes o letras. |
| `review_date` | Fecha de publicación. |
| `review_content` | Texto de la reseña profesional. |

## 3. Datasets procesados

Los archivos originales permanecen sin modificar. Los resultados reproducibles
se guardan en `processed_datasets/`.

### 3.1 IMDb: regulación bayesiana

El procesamiento conserva únicamente largometrajes con rating, incorpora los
nombres de sus directores y calcula una puntuación bayesiana:

$$
I_b = \frac{v}{v+m}R + \frac{m}{v+m}C
$$

- $R$: `averageRating` de la película.
- $v$: número de votos de la película.
- $C$: media global de las puntuaciones.
- $m$: percentil 90 del número de votos.

El ajuste acerca a la media las películas con pocos votos y deja que las más
votadas dependan principalmente de su nota real.

| Salida | Contenido |
|---|---|
| `imdb_peliculas.tsv` | Películas, títulos, año, directores, rating y votos. |
| `imdb_peliculas_bayesiano.tsv` | Mismos datos más `bayesianRating`, ordenados de mayor a menor. |

### 3.2 Rotten Tomatoes: normalización e imputación

El procesamiento elimina duplicados, corrige identificadores incompatibles y
normaliza las notas explícitas a una escala 0–10. Se admiten fracciones como
`4/5`, porcentajes como `85%` y calificaciones alfabéticas como `B+`.

Después, RoBERTa estima únicamente las notas realmente ausentes que tienen texto
y una clasificación Fresh/Rotten válida. Nunca sustituye una nota observada.

| Salida | Contenido |
|---|---|
| `rotten_tomatoes_movies_clean.csv` | Películas y campos necesarios para el ranking y el emparejamiento. |
| `rotten_tomatoes_critic_reviews_clean.csv` | Críticas originales más `normalized_score`. |
| `rotten_tomatoes_critic_reviews_scored.csv` | Añade `predicted_score`, `final_score`, `score_source` y trazabilidad del modelo. |

Las reseñas sin texto ni nota, o con una nota original de escala ambigua, quedan
como `unavailable`: se conservan para trazabilidad, pero no se usan para entrenar,
predecir ni calcular la media crítica. Los recuentos están en el apartado 5.3.

### 3.3 De las reseñas al ranking

Para cada película, la nota profesional es una **media ponderada de las reseñas
con puntuación disponible**, no el porcentaje del Tomatometer:

$$
C_f = \frac{\sum s_{observada}+\lambda\sum s_{predicha}}
{n_{observadas}+\lambda n_{predichas}},\qquad
n_f=n_{observadas}+\lambda n_{predichas}
$$

Cada nota observada pesa `1` y cada predicción pesa `λ = 0,8`. Ese descuento del
20 % reconoce que una estimación introduce error, conservando buena parte de la
información del texto. **No significa que el modelo tenga un 80 % de confianza**:
es un peso inicial que debe contrastarse con un análisis de sensibilidad.

La nota final es:

$$
S_f = I_b + \alpha\frac{n_f}{n_f+k}(C_f-I_b)
$$

Se usan `α = 0,30` y `k = 20`: la crítica puede aportar como máximo el 30 % del
peso final y su influencia crece con el número efectivo de reseñas; con 20 alcanza
el 15 %. El ajuste sube la nota si la crítica supera a IMDb y la baja si queda
por debajo. Sin críticas utilizables, `S_f = I_b`. Los tres parámetros son
configurables; expresan el criterio del ranking, no una calidad objetiva.

## 4. Modelo RoBERTa

El modelo base es
[`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base), un
Transformer preentrenado en inglés que se adapta mediante *fine-tuning* a una
tarea de regresión de notas cinematográficas.

### 4.1 Fine-tuning y elección del residual

Los experimentos comparables utilizaron 50.000 críticas para entrenamiento,
5.000 para validación y 5.000 para test, durante 2 épocas. La separación se hizo
por película para que las reseñas de un mismo título no apareciesen en
entrenamiento y test. Solo se usaron como etiquetas notas numéricas con una
escala explícita y se limitaron los textos a 256 tokens.

Se probaron dos objetivos:

1. **Regresión directa:** predecir la puntuación completa a partir del texto.
2. **Regresión residual:** partir de la media Fresh/Rotten y predecir únicamente
   cuánto se desvía la crítica respecto a esa referencia.

En el enfoque residual:

$$
r = y_{real}-\mu_{tipo}
$$

$$
\hat y = \operatorname{clip}(\mu_{tipo}+\hat r, 0, 10)
$$

Esto simplifica la tarea: `review_type` aporta la orientación general y RoBERTa
aprende principalmente la intensidad expresada en el texto.

### 4.2 Resultados comparables

| Método | MAE ↓ | RMSE ↓ |
|---|---:|---:|
| Baseline: media Fresh/Rotten | 1,044 | 1,324 |
| RoBERTa con regresión directa | 1,088 | 1,447 |
| **RoBERTa con regresión residual** | **0,884** | **1,134** |

El residual reduce el MAE un 15,3 % frente al baseline y un 18,7 % frente a la
regresión directa. Al agregar por película también mejora el MAE frente al
baseline: **0,637 frente a 0,801** sobre 1.191 películas de test. Por ello se
elige la regresión residual para completar las notas ausentes.

### 4.3 Entrenamiento final

Tras elegir el enfoque residual, el modelo `epac-roberta-residual-full` se
entrenó con todas las críticas utilizables:

| Métrica de entrenamiento | Valor |
|---|---:|
| Ejemplos | 677.353 |
| Fresh / Rotten | 431.920 / 245.433 |
| Épocas | 2 |
| `train_loss` | 5,180 |
| Duración | 17.640 s (4 h 54 min) |
| Rendimiento | 76,8 ejemplos/s |

Son métricas de entrenamiento, no una evaluación independiente del modelo full;
la elección del enfoque se basa en el test del experimento anterior.

## 5. Base de datos

PostgreSQL almacena las fuentes procesadas y el ranking. Las tablas se definen
con **SQLAlchemy** en [`src/db/tables/`](src/db/tables/) y sus cambios se aplican
mediante migraciones de **Alembic** en [`src/db/migrations/`](src/db/migrations/).

### 5.1 Tablas y relaciones

| Tabla | Columnas esenciales | Función |
|---|---|---|
| `imdb_movies` | `tconst` (PK), `primary_title`, `original_title`, `start_year`, `directors`, `average_rating`, `num_votes`, `bayesian_rating` | Películas y valoración del público ajustada. |
| `rt_movies` | `rt_id` (PK), `movie_title`, `original_release_date`, `directors`, `tomatometer_rating`, `tomatometer_count` | Catálogo RT y datos de emparejamiento. |
| `rt_reviews` | `review_id` (PK), `source_hash` único, `rotten_tomatoes_link` (FK), crítico/medio/fecha/tipo/texto, notas y procedencia del modelo | Reseñas sin sobrescribir las notas observadas; conserva también `unavailable_reason`. |
| `movie_matches` | PK compuesta `imdb_id` + `rt_id`, `match_method`, `match_confidence`, `verified` | Candidatos entre fuentes; como máximo un match verificado por ID de cada fuente. |
| `movie_ranking` | `imdb_id` (PK), `movie_title`, `rt_id` opcional, notas IMDb/crítica/final, recuentos observados/predichos/efectivos, parámetros, `rank_position`, `calculated_at` | Una fila por película IMDb, con título copiado de `primary_title` y entradas auditables de la fórmula. |

Cada película RT puede tener muchas reseñas. Solo los matches verificados aportan
críticas al ranking; las películas IMDb sin match permanecen en él. En
`rt_reviews` se separan `normalized_score`, `predicted_score`, `final_score` y
`score_source`, junto con modelo, versión y ejecución de inferencia.

### 5.2 Loaders

Los scripts de [`src/db/loaders/`](src/db/loaders/) validan columnas, formatos y
rangos y cargan por lotes. Los loaders de fuentes hacen *upsert*: insertan o
actualizan por ID —por `source_hash` en reseñas—, permitiendo repetir una carga.

| Script | Función |
|---|---|
| `load_imdb_movies.py` | Carga el TSV bayesiano; usa el título original si falta el principal y omite filas sin ambos. |
| `load_rt_movies.py` | Carga las películas RT antes de sus reseñas. |
| `load_rt_reviews.py` | Carga notas observadas, predichas y no disponibles, registrando la ejecución del modelo. |
| `load_movie_matches.py` | Empareja por título normalizado y año (hasta ±1); sin fecha RT exige título y director. Guarda ambiguos sin verificar y exporta los casos sin candidato. |
| `load_movie_ranking.py` | Agrega reseñas, aplica la fórmula y sustituye el ranking en una sola transacción, con posiciones deterministas. |

### 5.3 Resultados y registros omitidos

Cifras de la última carga comprobada, contrastadas con los archivos locales.
Son una instantánea del dataset, no contadores que se actualicen en este README.

| Resultado | Cantidad | Qué significa |
|---|---:|---|
| Películas IMDb cargadas | 350.680 | De 350.682 procesadas, se omiten 2 sin título principal ni original. |
| Películas RT cargadas | 17.712 | Se conserva todo el catálogo limpio. |
| Reseñas RT preparadas para la carga | 1.010.546 | De 1.130.017 originales se eliminan 119.471 duplicados exactos en el procesamiento. El CSV final tiene un `source_hash` distinto por reseña. |
| Reseñas con nota observada | 735.482 | Notas originales normalizadas a 0–10. |
| Reseñas con nota predicha | 273.215 | Notas ausentes completadas con RoBERTa. |
| Reseñas sin nota final | 1.849 | 390 sin nota ni texto y 1.459 con escala original no interpretable. Se almacenan, pero se excluyen de la media. |
| Parejas candidatas / verificadas | 16.695 / 16.024 | Solo las verificadas se incorporan al cálculo. |
| Películas RT sin candidato | 1.553 | No cumplen las reglas de título, año y/o director; se listan en `processed_datasets/matches/unmatched_rt_movies.csv`. |
| Películas en el ranking | 350.680 | 16.024 con match RT verificado y 334.656 solo IMDb. |

El **99,82 %** de las reseñas preparadas tiene nota. Quedar sin match no elimina
una película RT de la base: impide incorporar sus críticas a una película IMDb.
Hay además 135 películas RT con candidatos pero sin match verificado; los casos
ambiguos o con conflictos de unicidad quedan pendientes de revisión.

Los loaders imprimen filas leídas, escritas y omitidas. «Escritas» cuenta las
filas enviadas al *upsert*, no necesariamente nuevas inserciones; para conocer el
total persistido debe consultarse `COUNT(*)`. El recuento de reseñas anterior se
ha verificado en el CSV, sin una nueva consulta disponible a PostgreSQL.

## 6. Limitaciones del proyecto

- **Cobertura y sesgos:** IMDb refleja a sus usuarios y está sujeto a uso no
  comercial; RT es una instantánea con cobertura desigual por época y película.
  Las reseñas sin nota pueden tener un perfil diferente al de las etiquetadas.
- **Normalización:** convertir escalas distintas a 0–10, especialmente letras,
  aproxima criterios que no son idénticos entre críticos.
- **Modelo:** RoBERTa trabaja principalmente en inglés y con 256 tokens; puede
  perder contexto, interpretar mal la ironía y suavizar puntuaciones extremas.
- **Emparejamiento:** remakes, títulos repetidos y fechas/directores incompletos
  dejan casos ambiguos que necesitan revisión manual.
- **Alcance:** todas las películas IMDb cargadas pueden aparecer, pero solo las
  enlazadas reciben el ajuste RT. Esa cobertura desigual afecta a la comparación;
  el ranking combina opiniones, no mide una calidad objetiva.

## 7. Scripts y modelos

Las rutas siguientes son relativas a la raíz; los `__init__.py` organizan los
paquetes y no requieren ejecución.

| Paquete | Scripts y responsabilidad |
|---|---|
| `src/imdb/` | `imdb_processment.py`: filtra largometrajes, añade directores y calcula la nota bayesiana. |
| `src/rt/` | `rt_processment.py`: limpia y normaliza; `train_roberta_regression.py`: experimento directo; `rt_roberta_train.py`: experimento residual; `roberta_evaluation.py`: evaluación agregada por película; `rt_roberta_train_full.py`: entrenamiento residual completo; `predict_missing_scores.py`: inferencia de notas ausentes. |
| `src/db/tables/` | `base.py`: metadatos y nombres de restricciones; `imdb_movie.py`, `rt_movie.py`, `rt_review.py`, `movie_match.py`, `movie_ranking.py`: modelos de las cinco tablas. |
| `src/db/` | `session.py`: crea la conexión desde `DATABASE_URL`; `migrations/env.py` y `migrations/versions/*.py`: configuración e historial de Alembic. |
| `src/db/loaders/` | Los cinco loaders se describen en 5.2; `common.py` reúne validación, hash de reseñas y *upsert*. |
| `src/db/matching/` | `normalization.py`: normaliza títulos y directores; `matcher.py`: genera y verifica candidatos. |
| `src/db/ranking/` | `calculator.py`: calcula las puntuaciones y ordena el ranking con desempates estables. |

Los pesos, tokenizadores y métricas están en `models/epac-roberta-regression/`,
`models/epac-roberta-residual/` y `models/epac-roberta-residual-full/`. Esta última
carpeta es la usada en inferencia e incluye `training_metadata.json`; los
checkpoints sirven para reanudar entrenamientos. Para el detalle operativo del
modelo, véase [`src/rt/README.md`](src/rt/README.md).

## 8. Reproducir el proyecto

### 8.1 Entorno y configuración

Requisitos: Python 3.12, Poetry y Docker Compose. Ejecuta desde la raíz:

```powershell
poetry install --no-root
New-Item -ItemType File -Path .env
notepad .env
```

Si `.env` ya existe, edítalo directamente. Ejemplo con valores de desarrollo
que debes sustituir y mantener coherentes con la URL:

```dotenv
POSTGRES_DB=films_rating
POSTGRES_USER=films_user
POSTGRES_PASSWORD=change_me_local
DATABASE_URL=postgresql+psycopg://films_user:change_me_local@localhost:5433/films_rating
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASSWORD=change_me_admin
```

`DATABASE_URL` es la variable que leen Alembic y los loaders; las demás configuran
PostgreSQL y pgAdmin en Compose. El archivo `.env` está excluido de Git. Compose
lo lee automáticamente; para los comandos Python puedes usar `dotenvx` global:

```powershell
winget install dotenvx
```

### 8.2 Preparar datos y completar notas

Descarga las fuentes de 2: los cuatro `.tsv.gz` de IMDb en
`original_datasets/imdb/` y los dos CSV RT en
`original_datasets/rottentomatoes/`. Los originales y procesados son locales.

```powershell
poetry run python src/imdb/imdb_processment.py
poetry run python src/rt/rt_processment.py
```

Para repetir los experimentos y el entrenamiento final, preferiblemente en GPU
(local o Kaggle):

```powershell
poetry run python src/rt/train_roberta_regression.py
poetry run python src/rt/rt_roberta_train.py
poetry run python src/rt/roberta_evaluation.py
poetry run python src/rt/rt_roberta_train_full.py --fp16
```

Puedes omitir el entrenamiento si ya dispones del artefacto completo en
`models/epac-roberta-residual-full/`. Para GPU necesitas PyTorch con CUDA
compatible con tu equipo; las dependencias de Poetry por sí solas no garantizan
ese soporte. Verifica el dispositivo y ejecuta la inferencia:

```powershell
poetry run python -c "import torch; print(torch.cuda.is_available())"
poetry run python src/rt/predict_missing_scores.py --validate-only
poetry run python src/rt/predict_missing_scores.py --device cuda
```

Sin GPU, usa `--device cpu` y omite `--fp16` al entrenar. Si trabajas en Kaggle,
copia de vuelta el artefacto completo y el CSV `rotten_tomatoes_critic_reviews_scored.csv`
a sus rutas de `models/` y `processed_datasets/rottentomatoes/`.

### 8.3 Crear la base y cargar el ranking

Con Docker iniciado, aplica las migraciones existentes y carga en este orden:

```powershell
docker compose up -d postgres pgadmin
dotenvx run -- poetry run alembic upgrade head
dotenvx run -- poetry run python -m src.db.loaders.load_imdb_movies
dotenvx run -- poetry run python -m src.db.loaders.load_rt_movies
dotenvx run -- poetry run python -m src.db.loaders.load_rt_reviews --prediction-run-id residual-full-001
dotenvx run -- poetry run python -m src.db.loaders.load_movie_matches
dotenvx run -- poetry run python -m src.db.loaders.load_movie_ranking
```

Asigna a `--prediction-run-id` un identificador de la ejecución que produjo el
CSV. No hace falta generar nuevas revisiones Alembic para reproducir el esquema.
Para cambiar los pesos, el último comando admite `--alpha 0.30 --k 20
--prediction-weight 0.8`.

pgAdmin está en **http://localhost:5050**. Registra el servidor con host
`postgres`, puerto `5432` y las credenciales del `.env`; desde Python local se usa
`localhost:5433`. En el Query Tool puedes consultar:

```sql
SELECT rank_position, movie_title, imdb_bayesian_rating, critic_score, final_score
FROM movie_ranking
ORDER BY rank_position
LIMIT 100;
```

### 8.4 Visualización prevista: Streamlit

La siguiente fase será una página local con ranking, búsqueda por título,
filtros y comparación de notas. Streamlit ya figura en las dependencias, pero
la aplicación todavía no está implementada. Se prevé servirla en
**http://localhost:8501** con este comando, una vez creado el archivo:

```powershell
dotenvx run -- poetry run streamlit run src/visualization/app.py
```
