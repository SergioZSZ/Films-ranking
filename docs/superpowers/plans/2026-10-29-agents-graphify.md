# Repository Agent Guidance and Graphify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create durable repository instructions for coding agents and a local, ignored Graphify knowledge graph of the project.

**Architecture:** A root `AGENTS.md` is the human-readable source of project intent, data rules, ranking mathematics, and operating constraints. Graphify adds its native Codex guidance to that document and builds local HTML, JSON, and Markdown graph artifacts under ignored `graphify-out/`.

**Tech Stack:** Markdown, Git ignore rules, Graphify 0.9.x, Python, NetworkX

---

### Task 1: Add project-specific agent guidance

**Files:**
- Create: `AGENTS.md`
- Modify: `.gitignore`
- Reference: `README.md`
- Reference: `bbdd.md`
- Reference: `docs/superpowers/specs/2026-10-29-agents-graphify-design.md`

- [x] **Step 1: Verify the guidance and ignore rule are absent**

```powershell
if (Test-Path AGENTS.md) { throw 'AGENTS.md already exists' }
git check-ignore graphify-out/graph.json
if ($LASTEXITCODE -eq 0) { throw 'graphify-out is unexpectedly ignored already' }
```

Expected: `AGENTS.md` is absent and `git check-ignore` does not match the graph path.

- [x] **Step 2: Create the root guidance document**

Create `AGENTS.md` with these sections and rules:

```markdown
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
```

- [x] **Step 3: Ignore local Graphify artifacts**

Append this exact entry to `.gitignore`, preserving every existing user line:

```gitignore

# Grafo local regenerable de Graphify
graphify-out/
```

- [x] **Step 4: Install Graphify's native Codex section**

```powershell
graphify codex install
```

Expected: Graphify appends or updates one `## graphify` section in `AGENTS.md` without replacing the project-specific content.

- [x] **Step 5: Validate guidance and ignore behavior**

```powershell
rg -n "Propósito del proyecto|bayesianRating|lambda = 0.8|## graphify" AGENTS.md
git check-ignore -v graphify-out/graph.json
```

Expected: all four guidance patterns are found, and `git check-ignore` attributes the match to the new `graphify-out/` rule.

### Task 2: Build and validate the local knowledge graph

**Files:**
- Create locally: `graphify-out/graph.json`
- Create locally: `graphify-out/GRAPH_REPORT.md`
- Create locally: `graphify-out/graph.html`
- Create locally: `graphify-out/cost.json`

- [x] **Step 1: Resolve Graphify's Python interpreter and detect the corpus**

Use the installed `graphify` executable or `uv tool run --from graphifyy` to resolve the environment's Python, then write its absolute path to `graphify-out/.graphify_python` and the repository root to `graphify-out/.graphify_root`. Run `graphify.detect.detect(Path('.'))`, save the result as UTF-8 JSON in `graphify-out/.graphify_detect.json`, and print only total files, total words, category counts, and skipped-sensitive paths.

Expected: at least one code file and the repository documentation are detected. If detection exceeds 500 files or 2,000,000 words, stop and apply Graphify's mandatory corpus-narrowing gate before extraction.

- [x] **Step 2: Extract code and semantic content**

Run Graphify AST extraction for every detected code file. For detected documents, papers, and images, check the semantic cache with the installed Graphify extraction specification. If neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` exists, dispatch the uncached semantic files to general-purpose subagents in Graphify chunks of 20–25 files, with images isolated, and require each chunk to write valid node/edge/hyperedge JSON under `graphify-out/`.

Expected: `graphify-out/.graphify_ast.json` and `graphify-out/.graphify_semantic.json` contain valid JSON; failed semantic chunks are reported rather than silently omitted.

- [x] **Step 3: Build, diagnose, and label the graph**

Merge AST and semantic extraction into `graphify-out/.graphify_extract.json`. Build an undirected graph with paths relative to the repository root, cluster it, calculate cohesion, god nodes, surprising connections, and suggested questions, and export `graphify-out/graph.json` plus `graphify-out/GRAPH_REPORT.md`. Run Graphify's extraction diagnostic and surface every dangling, missing-endpoint, self-loop, or collapsed-edge warning. Assign each community a concise 2–5 word label and regenerate the graph/report with those labels.

Expected: the graph has at least one node, community labels contain no `Community N` placeholders, and diagnostics are recorded for the final report.

- [x] **Step 4: Export HTML and persist incremental metadata**

```powershell
graphify export html
```

Save Graphify's manifest and cumulative token cost, then remove only temporary `.graphify_*` extraction/chunk sidecars specified by the Graphify runbook. Do not delete `graph.json`, `GRAPH_REPORT.md`, `graph.html`, `cost.json`, `.graphify_python`, or `.graphify_root`.

- [x] **Step 5: Verify the generated outputs remain local**

```powershell
$required = @('graphify-out/graph.json', 'graphify-out/GRAPH_REPORT.md', 'graphify-out/graph.html', 'graphify-out/cost.json')
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing Graphify outputs: $($missing -join ', ')" }
git check-ignore graphify-out/graph.json graphify-out/GRAPH_REPORT.md graphify-out/graph.html
graphify diagnose multigraph --graph graphify-out/graph.json
```

Expected: all required outputs exist, all are ignored, and the diagnostic command completes with any integrity warnings made visible.

### Task 3: Review and commit repository-authored guidance

**Files:**
- Create: `AGENTS.md`
- Modify: `.gitignore`
- Create: `docs/superpowers/plans/2026-10-29-agents-graphify.md`

- [x] **Step 1: Review only the authored repository files**

```powershell
git diff --check -- AGENTS.md .gitignore docs/superpowers/plans/2026-10-29-agents-graphify.md
git status --short
```

Expected: no whitespace errors. Existing modifications to README, Compose, datasets, models, source, or environment files remain untouched.

- [x] **Step 2: Stage only authored changes**

Stage `AGENTS.md` and this plan normally. Because `.gitignore` already contains user modifications, construct and apply an index-only patch that adds only the Graphify ignore comment and rule relative to `HEAD`; do not stage the user's existing `.gitignore` additions.

Verify with:

```powershell
git diff --cached --check
git diff --cached --name-only
```

Expected staged paths: `.gitignore`, `AGENTS.md`, and `docs/superpowers/plans/2026-10-29-agents-graphify.md` only. The staged `.gitignore` diff contains only the Graphify comment/rule.

- [x] **Step 3: Commit the guidance**

```powershell
git commit -m "docs: add repository agent guidance"
```

Expected: one commit contains the agent guidance, isolated Graphify ignore rule, and implementation plan. `graphify-out/` remains ignored and uncommitted; all pre-existing user changes remain in the working tree.
