# AGENTS.md and Graphify design

## Objective

Create repository-level guidance for coding agents and generate a local Graphify
knowledge graph so future work can follow the project's data, modeling, and
scientific constraints without repeatedly rediscovering them.

## Repository guidance

Create a root `AGENTS.md` in Spanish. It will describe:

- the project goal: produce a historical film ranking from IMDb user ratings
  and Rotten Tomatoes professional reviews;
- the current pipelines under `src/imdb/` and `src/rt/`;
- the planned PostgreSQL tables documented in `bbdd.md`;
- the distinction between immutable source data, processed datasets, trained
  models, database state, and reproducible source code;
- the commands currently documented by the repository;
- scientific guardrails for leakage, score provenance, missing-score
  imputation, movie matching, and ranking validation;
- the instruction to preserve user changes in the dirty working tree and avoid
  committing datasets, secrets, database files, or generated Graphify output.

Graphify's native `agents install` command will add its standard section to the
same file. No post-commit hook will be installed.

## Ranking formula

The final score uses the Bayesian-adjusted IMDb user rating as its baseline:

\[
S_f = I_b + 0.30\frac{n_f}{n_f + 20}(C_f-I_b)
\]

Where:

- `I_b` is IMDb `bayesianRating`, not raw `averageRating`;
- `C_f` is the weighted mean of Rotten Tomatoes professional-review scores;
- `n_f` is the effective number of professional reviews;
- the maximum professional-review influence is `alpha = 0.30`;
- the confidence saturation parameter is `k = 20`.

Observed normalized critic scores have weight `1.0`. Scores imputed by RoBERTa
have provisional weight `lambda = 0.8`, applied both to the mean and to the
effective count:

\[
C_f = \frac{\sum s_{observed} + 0.8\sum s_{predicted}}
{n_{observed} + 0.8n_{predicted}}
\]

\[
n_f = n_{observed} + 0.8n_{predicted}
\]

The value `lambda = 0.8` must remain configurable and should be validated with
sensitivity analysis rather than treated as empirically final.

## Data provenance and quality rules

- Never overwrite an observed normalized score with a model prediction.
- Store score value, score source, and model version or run metadata separately.
- Split model train, validation, and test data by movie to avoid review leakage.
- Keep the final-score inputs available so every ranking value can be audited.
- Do not silently match IMDb and Rotten Tomatoes records by title alone; use
  title, release year or date, directors, confidence, and manual verification
  for ambiguous matches.
- Validate that all normalized, predicted, mean, Bayesian, and final scores stay
  in the `[0, 10]` range.

## Graphify execution

Run a full Graphify build from the repository root after `AGENTS.md` is ready so
the generated corpus includes the new project guidance. Generate the default
interactive HTML, JSON graph, and report under `graphify-out/`.

`graphify-out/` is a reproducible local artifact and will be added to
`.gitignore`. Graphify output must not be committed. Future documentation or
image changes require a manual Graphify update; no Git hook will be installed.

## Validation

- Confirm `AGENTS.md` contains both the project guidance and Graphify's native
  section.
- Confirm Git ignores `graphify-out/`.
- Run Graphify's graph health diagnostics and report integrity warnings.
- Confirm `graph.html`, `graph.json`, and `GRAPH_REPORT.md` exist locally.
- Preserve all pre-existing working-tree changes and commit only the explicitly
  authored repository guidance and ignore rule.
