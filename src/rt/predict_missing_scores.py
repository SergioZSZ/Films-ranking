"""Predice notas realmente ausentes con el RoBERTa residual completo."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RT_DATA_DIR = PROJECT_DIR / "processed_datasets" / "rottentomatoes"
INPUT_CSV = RT_DATA_DIR / "rotten_tomatoes_critic_reviews_clean.csv"
MODEL_DIR = PROJECT_DIR / "models" / "epac-roberta-residual-full"
OUTPUT_CSV = RT_DATA_DIR / "rotten_tomatoes_critic_reviews_scored.csv"
DEFAULT_BATCH_SIZE = 16

EXPECTED_OBJECTIVE = "normalized_score - mean(review_type)"
EXPECTED_INPUT_TEMPLATE = (
    "Sentiment classification: {review_type}. Review: {review_content}"
)
EXPECTED_POSTPROCESSING = (
    "clip(mean(review_type) + predicted_residual, 0, 10)"
)
REQUIRED_COLUMNS = {
    "review_score",
    "normalized_score",
    "review_content",
    "review_type",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye scores ausentes con el modelo residual RoBERTa completo."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="CSV limpio")
    parser.add_argument("--model", type=Path, default=MODEL_DIR, help="Artefacto Hugging Face")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="CSV de salida")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Tamaño de batch (por defecto: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Dispositivo de inferencia (por defecto: auto)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Predice solo las primeras N filas elegibles para una prueba corta",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida artefacto, CSV y recuentos sin cargar pesos ni escribir salida",
    )
    return parser.parse_args(argv)


def validate_model_artifact(model_dir: Path) -> dict:
    """Validate the saved full-residual model contract and return its metadata."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio del modelo: {model_dir}")

    required_files = {
        "config.json",
        "tokenizer_config.json",
        "training_metadata.json",
    }
    missing = sorted(name for name in required_files if not (model_dir / name).is_file())
    if not any((model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append("weights (model.safetensors o pytorch_model.bin)")
    has_tokenizer = (model_dir / "tokenizer.json").is_file() or all(
        (model_dir / name).is_file() for name in ("vocab.json", "merges.txt")
    )
    if not has_tokenizer:
        missing.append("tokenizer.json o vocab.json + merges.txt")
    if missing:
        raise FileNotFoundError(
            f"Artefacto de modelo incompleto en {model_dir}; faltan: {', '.join(missing)}"
        )

    metadata_path = model_dir / "training_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"No se puede leer metadata valida de {metadata_path}: {exc}") from exc

    expected_fields = {
        "status": "completed",
        "objective": EXPECTED_OBJECTIVE,
        "input_template": EXPECTED_INPUT_TEMPLATE,
        "postprocessing": EXPECTED_POSTPROCESSING,
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(
                f"Metadata incompatible: {field} debe ser {expected!r}, recibido {actual!r}"
            )

    means = metadata.get("means")
    if not isinstance(means, dict):
        raise ValueError("Metadata incompatible: means debe contener Fresh y Rotten")
    for review_type in ("Fresh", "Rotten"):
        value = means.get(review_type)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Metadata incompatible: means[{review_type!r}] no es finita")

    max_length = metadata.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("Metadata incompatible: max_length debe ser un entero positivo")
    return metadata


def load_reviews(input_csv: Path) -> pd.DataFrame:
    """Load the cleaned review CSV and validate its inference-facing fields."""
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de entrada: {input_csv}")
    reviews = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(reviews.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {', '.join(missing)}")

    normalized = reviews["normalized_score"].str.strip()
    populated = normalized.ne("")
    numeric = pd.to_numeric(normalized.where(populated), errors="coerce")
    invalid = populated & (numeric.isna() | ~numeric.between(0, 10))
    if invalid.any():
        examples = ", ".join(normalized[invalid].head(3).tolist())
        raise ValueError(
            "normalized_score debe ser numerico y estar en [0, 10]; "
            f"valores invalidos de ejemplo: {examples}"
        )
    return reviews


def _canonical_review_types(reviews: pd.DataFrame) -> pd.Series:
    return reviews["review_type"].astype("string").str.strip().str.title()


def classify_rows(reviews: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Classify score provenance and identify rows eligible for inference."""
    normalized = reviews["normalized_score"].astype("string").str.strip()
    raw_score = reviews["review_score"].astype("string").str.strip()
    content = reviews["review_content"].astype("string").str.strip()
    review_types = _canonical_review_types(reviews)

    has_normalized = normalized.ne("")
    has_raw_score = raw_score.ne("")
    has_text = content.ne("")
    valid_type = review_types.isin(("Fresh", "Rotten"))
    eligible = ~has_normalized & ~has_raw_score & has_text & valid_type

    sources = pd.Series("pending_prediction_limit", index=reviews.index, dtype="string")
    sources.loc[has_normalized] = "normalized_original"
    sources.loc[~has_normalized & has_raw_score] = "unavailable_unparseable"
    sources.loc[~has_normalized & ~has_raw_score & ~has_text] = (
        "unavailable_missing_text"
    )
    sources.loc[~has_normalized & ~has_raw_score & has_text & ~valid_type] = (
        "unavailable_invalid_review_type"
    )
    return sources, eligible


def build_model_texts(
    reviews: pd.DataFrame,
    indices: np.ndarray,
    template: str = EXPECTED_INPUT_TEMPLATE,
) -> list[str]:
    """Render the exact text template used to train the residual model."""
    positions = np.asarray(indices, dtype=int)
    selected = reviews.iloc[positions]
    review_types = _canonical_review_types(selected)
    content = selected["review_content"].astype("string").str.strip()
    return [
        template.format(review_type=review_type, review_content=review_content)
        for review_type, review_content in zip(review_types, content, strict=True)
    ]


def reconstruct_scores(
    residuals: np.ndarray,
    review_types: list[str] | pd.Series,
    means: dict[str, float],
) -> np.ndarray:
    """Convert residual model outputs into absolute scores in the 0-10 range."""
    residual_array = np.asarray(residuals, dtype=float).reshape(-1)
    if not np.isfinite(residual_array).all():
        raise ValueError("El modelo produjo predicciones residuales no finitas")

    canonical_types = [str(value).strip().title() for value in review_types]
    if len(canonical_types) != len(residual_array):
        raise ValueError("El numero de residuos no coincide con los review_type")
    try:
        baselines = np.asarray([means[value] for value in canonical_types], dtype=float)
    except KeyError as exc:
        raise ValueError(f"review_type sin media residual: {exc.args[0]}") from exc
    scores = baselines + residual_array
    if not np.isfinite(scores).all():
        raise ValueError("La reconstruccion genero predicciones no finitas")
    return np.clip(scores, 0.0, 10.0)


def assemble_output(
    reviews: pd.DataFrame,
    predicted_indices: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    metadata_identifier: str,
) -> pd.DataFrame:
    """Return a copy of the reviews with final score and provenance columns."""
    output = reviews.copy()
    sources, eligible = classify_rows(output)
    positions = np.asarray(predicted_indices, dtype=int)
    score_values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(positions) != len(score_values):
        raise ValueError("El numero de indices y predicciones no coincide")
    if len(set(positions.tolist())) != len(positions):
        raise ValueError("Los indices de prediccion no pueden estar duplicados")
    if len(positions) and (
        positions.min() < 0
        or positions.max() >= len(output)
        or not eligible.iloc[positions].all()
    ):
        raise ValueError("Solo se pueden asignar predicciones a filas elegibles")
    if not np.isfinite(score_values).all():
        raise ValueError("No se pueden guardar predicciones no finitas")

    output["predicted_score"] = pd.Series("", index=output.index, dtype="string")
    output["final_score"] = pd.Series("", index=output.index, dtype="string")
    output["score_source"] = sources
    output["prediction_model"] = pd.Series("", index=output.index, dtype="string")
    output["prediction_metadata"] = pd.Series("", index=output.index, dtype="string")

    normalized = output["normalized_score"].astype("string").str.strip()
    observed = sources.eq("normalized_original")
    output.loc[observed, "final_score"] = normalized.loc[observed]

    if len(positions):
        labels = output.index[positions]
        formatted = pd.Series(
            [f"{value:.4f}" for value in score_values], index=labels, dtype="string"
        )
        output.loc[labels, "predicted_score"] = formatted
        output.loc[labels, "final_score"] = formatted
        output.loc[labels, "score_source"] = "roberta_prediction"
        output.loc[labels, "prediction_model"] = model_name
        output.loc[labels, "prediction_metadata"] = metadata_identifier
    return output


def _print_source_counts(sources: pd.Series) -> None:
    print("\n=== FILAS POR PROCEDENCIA ===")
    print(sources.value_counts().to_string())


def _resolve_device(requested: str, torch_module) -> str:
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False")
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


def predict_selected_rows(
    reviews: pd.DataFrame,
    selected: np.ndarray,
    model_dir: Path,
    metadata: dict,
    batch_size: int,
    requested_device: str,
) -> np.ndarray:
    """Load the local model and predict absolute scores for selected row positions."""
    if len(selected) == 0:
        return np.asarray([], dtype=float)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _resolve_device(requested_device, torch)
    print(f"Dispositivo de inferencia: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.to(device).eval()

    predicted_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            batch_indices = selected[offset : offset + batch_size]
            texts = build_model_texts(
                reviews,
                batch_indices,
                template=metadata["input_template"],
            )
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=metadata["max_length"],
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            residuals = model(**inputs).logits.reshape(-1).float().cpu().numpy()
            review_types = _canonical_review_types(reviews.iloc[batch_indices]).tolist()
            predicted_batches.append(
                reconstruct_scores(residuals, review_types, metadata["means"])
            )

            completed = min(offset + batch_size, len(selected))
            batch_number = offset // batch_size
            if batch_number % 250 == 0 or completed == len(selected):
                print(f"Predicciones: {completed:,}/{len(selected):,}")
    return np.concatenate(predicted_batches)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = validate_model_artifact(args.model)
    reviews = load_reviews(args.input)
    sources, eligible = classify_rows(reviews)
    eligible_indices = np.flatnonzero(eligible.to_numpy())
    selected = eligible_indices[: args.limit] if args.limit else eligible_indices

    print(f"Filas totales: {len(reviews):,}")
    print(f"Elegibles para RoBERTa: {len(eligible_indices):,}")
    print(f"Seleccionadas en esta ejecución: {len(selected):,}")
    _print_source_counts(sources)
    if args.validate_only:
        print("\nValidación completada; no se cargaron pesos ni se escribió salida.")
        return

    predictions = predict_selected_rows(
        reviews=reviews,
        selected=selected,
        model_dir=args.model,
        metadata=metadata,
        batch_size=args.batch_size,
        requested_device=args.device,
    )
    output = assemble_output(
        reviews=reviews,
        predicted_indices=selected,
        predictions=predictions,
        model_name=args.model.name,
        metadata_identifier="training_metadata.json",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8")

    print("\n=== RESULTADO ===")
    print(output["score_source"].value_counts().to_string())
    print(f"Archivo generado: {args.output}")
    print("Las predicciones son estimaciones, no notas originales de Rotten Tomatoes.")


if __name__ == "__main__":
    main()
