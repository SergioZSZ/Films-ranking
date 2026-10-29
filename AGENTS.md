# AGENTS.md

## Propósito del proyecto

Este repositorio construye un ranking histórico de películas combinando dos
señales distintas:

- IMDb aporta valoraciones de usuarios normales. `averageRating` no representa
  crítica profesional.
- Rotten Tomatoes aporta reseñas de críticos de cine. Sus notas explícitas se
  normalizan a 0–10 y las notas ausentes se imputan con RoBERTa.

El resultado debe ser reproducible, auditable y trazable desde cada nota final
hasta los datos observados o predichos que la originaron.

## Flujo de datos

1. Los archivos originales permanecen inmutables en `original_datasets/`.
2. `src/imdb/imdb_processment.py` filtra películas, incorpora directores y
   genera `averageRating`, `numVotes` y `bayesianRating` en
   `processed_datasets/imdb/`.
3. `src/rt/rt_processment.py` reduce los CSV de Rotten Tomatoes y normaliza las
   notas explícitas de críticos a una escala 0–10.
4. El modelo RoBERTa residual rellena únicamente notas ausentes que tengan texto
   utilizable; nunca sustituye una nota observada.
5. PostgreSQL almacena películas, ratings, reseñas, procedencia de las notas y
   correspondencias IMDb–Rotten Tomatoes. El esquema previsto está en `bbdd.md`.
6. Las películas se emparejan entre fuentes antes de calcular el ranking final.

## Fórmula del ranking

La base es la puntuación bayesiana de IMDb, no `averageRating` sin ajustar:

`S_f = I_b + 0.30 * (n_f / (n_f + 20)) * (C_f - I_b)`

- `I_b`: `bayesianRating` de IMDb.
- `C_f`: media ponderada de notas profesionales en escala 0–10.
- `n_f`: número efectivo de críticas profesionales.
- `alpha = 0.30`: influencia profesional máxima.
- `k = 20`: parámetro de saturación de confianza.

Una nota profesional observada pesa `1.0`. Una nota imputada por RoBERTa pesa
provisionalmente `lambda = 0.8`:

`C_f = (sum(observed) + 0.8 * sum(predicted)) / (n_observed + 0.8 * n_predicted)`

`n_f = n_observed + 0.8 * n_predicted`

Mantén `alpha`, `k` y `lambda` configurables. `lambda = 0.8` es una decisión
inicial que requiere análisis de sensibilidad, no una constante validada.

## Reglas científicas y de calidad

- Conserva por separado `normalized_score`, `predicted_score`, `final_score` y
  `score_source`; no sobrescribas observaciones con predicciones.
- Registra la versión o ejecución del modelo usada para cada imputación.
- Divide train, validación y test por película para impedir fuga entre reseñas
  de la misma película.
- Ajusta medias y parámetros únicamente con la partición de entrenamiento.
- Evalúa RoBERTa frente al baseline Fresh/Rotten con MAE y RMSE, pero valida
  también el efecto sobre ordenación y estabilidad del ranking.
- Comprueba que todas las notas normalizadas, predichas, agregadas, bayesianas y
  finales estén en `[0, 10]`.
- No unas fuentes únicamente por título. Usa título normalizado, año o fecha de
  estreno y directores; conserva `match_method`, `match_confidence` y
  `verified`, y revisa manualmente los casos ambiguos.
- Haz que cada puntuación final conserve sus entradas (`I_b`, `C_f`, recuentos,
  pesos y parámetros) para poder auditarla.

## Estructura relevante

- `src/imdb/`: procesamiento y regulación bayesiana de IMDb.
- `src/rt/`: normalización, entrenamiento, evaluación e inferencia RoBERTa.
- `models/`: artefactos locales de modelos y métricas; evita versionar pesos o
  checkpoints grandes.
- `original_datasets/`: datos descargados sin modificar; no se versionan.
- `processed_datasets/`: resultados reproducibles; no se versionan salvo una
  decisión explícita.
- `bbdd.md`: diseño previsto de tablas.
- `docker-compose.yml`: PostgreSQL y pgAdmin para desarrollo local.
- `graphify-out/`: grafo regenerable y estrictamente local.

## Comandos habituales

Ejecuta desde la raíz del repositorio:

```powershell
poetry install
poetry run python src/imdb/imdb_processment.py
poetry run python src/rt/rt_processment.py
poetry run python src/rt/rt_roberta_train_full.py --help
docker compose config
docker compose up -d postgres pgadmin
```

No entrenes modelos completos, descargues datasets ni levantes servicios Docker
salvo que la tarea lo requiera expresamente. Prefiere validaciones pequeñas y
deterministas durante el desarrollo.

## Convenciones de trabajo

- Usa Python 3.12 y sigue la configuración de `pyproject.toml`/Poetry.
- Trabaja con `pathlib.Path`, UTF-8 y rutas relativas a la raíz del proyecto.
- Añade funciones pequeñas, type hints y docstrings cuando aclaren contratos.
- Valida columnas obligatorias, duplicados, nulos y rangos antes de escribir
  archivos o insertar registros.
- No incluyas secretos ni credenciales reales. Usa variables de entorno y
  conserva solamente valores locales de desarrollo en Compose.
- No modifiques datasets originales ni artefactos del usuario.
- El árbol de trabajo puede contener cambios ajenos: inspecciona `git status`,
  limita los parches a la tarea y no descartes ni reformatees cambios existentes.
- Actualiza README, `bbdd.md` y pruebas cuando cambien contratos, fórmulas o
  esquemas.

## Criterios de finalización

Antes de declarar una tarea terminada:

1. Ejecuta las pruebas o validaciones específicas del cambio.
2. Comprueba formatos, rangos, duplicados y procedencia de datos afectados.
3. Revisa `git diff --check` y el diff limitado a los archivos tocados.
4. Indica qué no se ejecutó, especialmente entrenamientos, datasets grandes o
   servicios Docker.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
