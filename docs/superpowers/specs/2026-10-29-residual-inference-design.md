# Full residual RoBERTa inference design

## Objective

Adapt `src/rt/predict_missing_scores.py` so it can consume the completed
`models/epac-roberta-residual-full` artifact returned from Kaggle and predict
only genuinely missing Rotten Tomatoes critic scores.

The inference pipeline must reproduce the full-training input and output
contract exactly. It must never replace an explicit critic score merely because
that score could not be normalized.

## Inputs and outputs

Default paths are relative to the repository root:

- Input CSV:
  `processed_datasets/rottentomatoes/rotten_tomatoes_critic_reviews_clean.csv`
- Model directory: `models/epac-roberta-residual-full`
- Output CSV:
  `processed_datasets/rottentomatoes/rotten_tomatoes_critic_reviews_scored.csv`

Command-line arguments allow overriding the input, model, output, batch size,
device, and an optional prediction limit for smoke tests. A validation-only
flag checks paths, metadata, columns, and target counts without loading model
weights. Device selection is automatic by default and accepts explicit CPU or
CUDA selection.

## Model contract

The model directory must contain the Hugging Face configuration, weights,
tokenizer files, and `training_metadata.json`. Inference fails before loading
the dataset or model when the artifact is incomplete or incompatible.

Required metadata:

- `status` equals `completed`;
- `objective` equals `normalized_score - mean(review_type)`;
- `input_template` equals
  `Sentiment classification: {review_type}. Review: {review_content}`;
- `postprocessing` describes adding the review-type mean and clipping to
  `[0, 10]`;
- `means` contains finite numeric values for both `Fresh` and `Rotten`;
- `max_length` is a positive integer.

The metadata file is the source of truth for the Fresh/Rotten means and maximum
token length. Inference must not hard-code the means from an earlier experiment.

## Eligibility and provenance

Each row receives exactly one `score_source` classification:

- `normalized_original`: `normalized_score` is a valid number in `[0, 10]`.
- `roberta_prediction`: `review_score` is empty, `normalized_score` is empty,
  review text is non-empty, and `review_type` is `Fresh` or `Rotten`.
- `unavailable_unparseable`: `review_score` is non-empty while
  `normalized_score` is empty. These explicit but ambiguous scores are neither
  training data nor inference targets.
- `unavailable_missing_text`: the original score is empty and no usable review
  text exists.
- `unavailable_invalid_review_type`: the original and normalized scores are
  empty and text exists, but the review type cannot select a residual baseline.
- `pending_prediction_limit`: the row is eligible for inference but was not
  selected because a smoke-test prediction limit was applied.

Only rows eligible to become `roberta_prediction` are tokenized and sent through
the model. An optional limit restricts this set for a smoke test;
limited-but-not-predicted rows remain `pending_prediction_limit` rather than
being mislabeled unavailable.

The output preserves every input column and adds or replaces:

- `predicted_score`: reconstructed prediction for inferred rows only;
- `final_score`: normalized original when available, otherwise the prediction;
- `score_source`: provenance classification;
- `prediction_model`: model directory name for inferred rows;
- `prediction_metadata`: metadata filename or stable model-run identifier for
  inferred rows.

## Residual inference

For each eligible row, construct the exact training text:

```text
Sentiment classification: {review_type}. Review: {review_content}
```

The raw model output is a residual. Reconstruct the absolute critic score with:

```text
predicted_score = clip(means[review_type] + residual, 0, 10)
```

Predictions are emitted with four decimal places. Batches run under
`torch.inference_mode()`, and progress is printed without logging review text.

## Failure behavior

The command fails with an actionable error when:

- the input CSV, model directory, required model files, or metadata are absent;
- metadata fields are missing or incompatible;
- required CSV columns are missing;
- an explicit CUDA device is requested but unavailable;
- normalized values fall outside `[0, 10]`;
- reconstructed predictions are non-finite.

The input CSV is never modified. The final output is written only after all
selected batches complete successfully.

## Testing and documentation

Pure functions cover metadata validation, row classification, training-text
construction, residual reconstruction, and output-column assembly. Unit tests
use small in-memory frames and synthetic residuals, so they do not download or
load RoBERTa weights.

`src/rt/README.md` documents the Kaggle artifact layout, validation-only or
smoke-test command, full command, output columns, and the fact that explicit but
unparseable scores remain unavailable.

The full 273,000-plus-row inference is outside this change because the completed
Kaggle model is not yet present locally. Verification uses unit tests, CLI help,
syntax checks, and a metadata-validation fixture.
