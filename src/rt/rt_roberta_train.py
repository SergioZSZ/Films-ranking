"""
EPAC — RoBERTa-base con regresión residual.

Objetivo:
    Predecir puntuaciones de críticas cinematográficas
    en una escala de 0 a 10.

Entradas:
    - review_content: texto de la crítica.
    - review_type: Fresh / Rotten.

Target:
    residual = normalized_score - mean(review_type)

Predicción:
    predicted_score = mean(review_type) + predicted_residual

Las medias se calculan exclusivamente sobre entrenamiento.

La separación train/validation/test se realiza por película
para evitar fugas de información.

Los resultados se guardan en un directorio independiente
del primer experimento.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =========================== CONFIGURACIÓN ===========================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = (
    PROJECT_DIR
    / "processed_datasets"
    / "rottentomatoes"
    / "rotten_tomatoes_critic_reviews_clean.csv"
)

# Directorio independiente del experimento anterior.
MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "epac-roberta-residual"
)

BASE_MODEL = "FacebookAI/roberta-base"

SEED = 42

MAX_TOKENS = 256

MAX_TRAIN = 50_000
MAX_VALID = 5_000
MAX_TEST = 5_000

EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

EARLY_STOPPING_PATIENCE = 1

VALID_REVIEW_TYPES = ("Fresh", "Rotten")

# =====================================================================


FRACTION_PATTERN = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)\s*$"
)


# =========================== CARGA DE DATOS ===========================

def load_reliable_examples(path: Path) -> pd.DataFrame:
    """
    Carga críticas que tienen:
        - Identificador de película.
        - Texto.
        - Clasificación Fresh/Rotten.
        - Puntuación numérica con escala explícita.

    Utilizamos únicamente fracciones para evitar conversiones
    arbitrarias de puntuaciones alfabéticas.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el dataset: {path}"
        )

    columns = [
        "rotten_tomatoes_link",
        "review_score",
        "review_type",
        "review_content",
    ]

    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
    )

    for column in columns:
        frame[column] = frame[column].str.strip()

    # Mantener únicamente críticas con información suficiente.
    valid_rows = (
        frame["rotten_tomatoes_link"].ne("")
        & frame["review_content"].ne("")
        & frame["review_type"].isin(VALID_REVIEW_TYPES)
    )

    frame = frame.loc[valid_rows].copy()

    # Seleccionar puntuaciones con denominador explícito.
    matches = frame["review_score"].str.match(
        FRACTION_PATTERN
    )

    frame = frame.loc[matches].copy()

    parts = frame["review_score"].str.extract(
        FRACTION_PATTERN.pattern
    )

    numerator = pd.to_numeric(
        parts[0].str.replace(",", ".", regex=False),
        errors="coerce",
    )

    denominator = pd.to_numeric(
        parts[1].str.replace(",", ".", regex=False),
        errors="coerce",
    )

    valid_scores = (
        denominator.gt(0)
        & numerator.ge(0)
        & numerator.le(denominator)
    )

    frame = frame.loc[valid_scores].copy()

    # Normalizar puntuación a 0-10.
    frame["target"] = (
        10
        * numerator.loc[valid_scores]
        / denominator.loc[valid_scores]
    ).astype("float32")

    frame = frame.reset_index(drop=True)

    if frame.empty:
        raise ValueError(
            "No se encontraron ejemplos válidos."
        )

    return frame


# =========================== DIVISIÓN DE DATOS ===========================

def split_by_movie(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos por identificador de película.

    Conservamos el mismo procedimiento y semilla
    del primer experimento para permitir comparaciones.
    """

    groups = frame["rotten_tomatoes_link"]

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=SEED,
    )

    train_val_idx, test_idx = next(
        outer_split.split(frame, groups=groups)
    )

    train_val_frame = frame.iloc[train_val_idx]

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=1 / 9,
        random_state=SEED + 1,
    )

    train_rel, valid_rel = next(
        inner_split.split(
            train_val_frame,
            groups=train_val_frame["rotten_tomatoes_link"],
        )
    )

    train = frame.iloc[train_val_idx[train_rel]].copy()

    valid = frame.iloc[train_val_idx[valid_rel]].copy()

    test = frame.iloc[test_idx].copy()

    # Evitar textos exactamente iguales entre particiones.
    heldout_texts = (
        set(valid["review_content"])
        | set(test["review_content"])
    )

    train = train.loc[
        ~train["review_content"].isin(heldout_texts)
    ].copy()

    test = test.loc[
        ~test["review_content"].isin(
            set(valid["review_content"])
        )
    ].copy()

    # Verificar separación por película.
    assert set(
        train["rotten_tomatoes_link"]
    ).isdisjoint(
        valid["rotten_tomatoes_link"]
    )

    assert set(
        train["rotten_tomatoes_link"]
    ).isdisjoint(
        test["rotten_tomatoes_link"]
    )

    assert set(
        valid["rotten_tomatoes_link"]
    ).isdisjoint(
        test["rotten_tomatoes_link"]
    )

    def sample(
        subset: pd.DataFrame,
        limit: int,
    ) -> pd.DataFrame:

        if len(subset) > limit:
            subset = subset.sample(
                n=limit,
                random_state=SEED,
            )

        return subset.reset_index(drop=True)

    train = sample(train, MAX_TRAIN)
    valid = sample(valid, MAX_VALID)
    test = sample(test, MAX_TEST)

    for name, subset in [
        ("train", train),
        ("validation", valid),
        ("test", test),
    ]:
        if subset.empty:
            raise ValueError(
                f"La partición {name} está vacía."
            )

    return train, valid, test


# =========================== REGRESIÓN RESIDUAL ===========================

def prepare_residual_targets(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Calcula la puntuación media de cada categoría
    utilizando exclusivamente entrenamiento.

    Después calcula:

        residual = target - category_mean

    Finalmente prepara la entrada textual del modelo.
    """

    group_means = (
        train.groupby("review_type")["target"]
        .mean()
        .to_dict()
    )

    fallback_mean = float(
        train["target"].mean()
    )

    print("\nMedias calculadas sobre entrenamiento:")

    for review_type, mean in group_means.items():
        print(
            f"  {review_type}: {mean:.4f}/10"
        )

    def transform(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = frame.copy()

        # Media de la categoría Fresh/Rotten.
        frame["base_score"] = (
            frame["review_type"]
            .map(group_means)
            .fillna(fallback_mean)
            .astype("float32")
        )

        # Target residual.
        frame["residual"] = (
            frame["target"]
            - frame["base_score"]
        ).astype("float32")

        # Entrada del modelo con información estructurada
        # y texto de la crítica.
        frame["model_input"] = (
            "Sentiment classification: "
            + frame["review_type"]
            + ". Review: "
            + frame["review_content"]
        )

        return frame

    return (
        transform(train),
        transform(valid),
        transform(test),
        group_means,
        fallback_mean,
    )


# =========================== TOKENIZACIÓN ===========================

def as_dataset(
    frame: pd.DataFrame,
    tokenizer,
) -> Dataset:
    """
    Convierte las críticas en un Dataset de Hugging Face.

    El target de entrenamiento es el residual,
    no la puntuación absoluta.
    """

    data = pd.DataFrame({
        "model_input": frame["model_input"],
        "labels": frame["residual"].astype("float32"),
    })

    dataset = Dataset.from_pandas(
        data,
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["model_input"],
            truncation=True,
            max_length=MAX_TOKENS,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["model_input"],
    )


# =========================== EVALUACIÓN ===========================

def build_compute_metrics(
    validation_baselines: np.ndarray,
):
    """
    Calcula MAE y RMSE sobre puntuaciones absolutas.

    El modelo predice un residual, por lo que necesitamos
    sumar la media correspondiente antes de evaluar.
    """

    def compute_metrics(prediction):

        predicted_residuals = np.asarray(
            prediction.predictions
        ).reshape(-1)

        true_residuals = np.asarray(
            prediction.label_ids
        ).reshape(-1)

        if len(predicted_residuals) != len(validation_baselines):
            raise ValueError(
                "Las predicciones no coinciden con "
                "el tamaño de validación."
            )

        predicted_scores = np.clip(
            validation_baselines + predicted_residuals,
            0.0,
            10.0,
        )

        true_scores = (
            validation_baselines + true_residuals
        )

        return {
            "mae": float(
                mean_absolute_error(
                    true_scores,
                    predicted_scores,
                )
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        true_scores,
                        predicted_scores,
                    )
                )
            ),
        }

    return compute_metrics


# =========================== ENTRENAMIENTO ===========================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        print(
            "GPU disponible:",
            torch.cuda.get_device_name(0),
        )
    else:
        print(
            "AVISO: no se detecta GPU CUDA. "
            "El entrenamiento utilizará CPU."
        )

    # -------------------------------------------------
    # 1. Cargar ejemplos
    # -------------------------------------------------

    data = load_reliable_examples(INPUT_CSV)

    train_df, valid_df, test_df = split_by_movie(data)

    print(
        f"\nEjemplos disponibles: {len(data):,}"
    )

    print(
        f"Train/valid/test: "
        f"{len(train_df):,} / "
        f"{len(valid_df):,} / "
        f"{len(test_df):,}"
    )

    print(
        "La separación se ha realizado por película."
    )

    # -------------------------------------------------
    # 2. Preparar regresión residual
    # -------------------------------------------------

    (
        train_df,
        valid_df,
        test_df,
        group_means,
        fallback_mean,
    ) = prepare_residual_targets(
        train_df,
        valid_df,
        test_df,
    )

    # -------------------------------------------------
    # 3. Tokenizar
    # -------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
    )

    ds_train = as_dataset(train_df, tokenizer)
    ds_valid = as_dataset(valid_df, tokenizer)
    ds_test = as_dataset(test_df, tokenizer)

    # -------------------------------------------------
    # 4. Cargar RoBERTa
    # -------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=1,
            problem_type="regression",
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # 5. Configuración de entrenamiento
    # -------------------------------------------------

    args = TrainingArguments(

        output_dir=str(
            MODEL_DIR / "checkpoints"
        ),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="mae",
        greater_is_better=False,

        save_total_limit=2,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        num_train_epochs=EPOCHS,

        weight_decay=WEIGHT_DECAY,

        logging_steps=100,

        fp16=torch.cuda.is_available(),

        dataloader_pin_memory=torch.cuda.is_available(),

        report_to="none",

        seed=SEED,
        data_seed=SEED,
    )

    # -------------------------------------------------
    # 6. Métricas de validación
    # -------------------------------------------------

    validation_baselines = valid_df[
        "base_score"
    ].to_numpy(dtype=float)

    compute_metrics = build_compute_metrics(
        validation_baselines
    )

    # Early stopping a partir de 3 épocas.
    callbacks = []

    if EPOCHS >= 3:

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        )

    # -------------------------------------------------
    # 7. Crear Trainer
    # -------------------------------------------------

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=ds_train,

        eval_dataset=ds_valid,

        processing_class=tokenizer,

        compute_metrics=compute_metrics,

        callbacks=callbacks,
    )

    # -------------------------------------------------
    # 8. Entrenar
    # -------------------------------------------------

    print("\nIniciando entrenamiento residual...")

    trainer.train()

    # -------------------------------------------------
    # 9. Predicciones sobre TEST
    # -------------------------------------------------

    raw_residuals = (
        trainer.predict(ds_test)
        .predictions
        .reshape(-1)
    )

    baseline_scores = test_df[
        "base_score"
    ].to_numpy(dtype=float)

    predicted_scores = np.clip(
        baseline_scores + raw_residuals,
        0.0,
        10.0,
    )

    true_scores = test_df[
        "target"
    ].to_numpy(dtype=float)

    # -------------------------------------------------
    # 10. Evaluar
    # -------------------------------------------------

    mae_roberta = float(
        mean_absolute_error(
            true_scores,
            predicted_scores,
        )
    )

    rmse_roberta = float(
        np.sqrt(
            mean_squared_error(
                true_scores,
                predicted_scores,
            )
        )
    )

    mae_baseline = float(
        mean_absolute_error(
            true_scores,
            baseline_scores,
        )
    )

    rmse_baseline = float(
        np.sqrt(
            mean_squared_error(
                true_scores,
                baseline_scores,
            )
        )
    )

    # -------------------------------------------------
    # 11. Guardar métricas
    # -------------------------------------------------

    metrics = {

        "experiment": "roberta_residual",

        "model": BASE_MODEL,

        "seed": SEED,

        "max_tokens": MAX_TOKENS,

        "epochs": EPOCHS,

        "train_rows": len(train_df),

        "validation_rows": len(valid_df),

        "test_rows": len(test_df),

        "train_movies": int(
            train_df["rotten_tomatoes_link"].nunique()
        ),

        "validation_movies": int(
            valid_df["rotten_tomatoes_link"].nunique()
        ),

        "test_movies": int(
            test_df["rotten_tomatoes_link"].nunique()
        ),

        "test_mae_roberta": mae_roberta,

        "test_rmse_roberta": rmse_roberta,

        "test_mae_fresh_rotten_baseline": mae_baseline,

        "test_rmse_fresh_rotten_baseline": rmse_baseline,

        "train_group_means": {
            key: float(value)
            for key, value in group_means.items()
        },

        "fallback_mean": fallback_mean,

        "prediction_method": (
            "score = clip("
            "mean(review_type) + roberta_residual, "
            "0, 10)"
        ),

        "input_format": (
            "Sentiment classification: {review_type}. "
            "Review: {review_content}"
        ),

        "training_labels": (
            "only numeric fractions "
            "with explicit denominator"
        ),

        "source_csv": str(INPUT_CSV),
    }

    # -------------------------------------------------
    # 12. Guardar modelo
    # -------------------------------------------------

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    metrics_path = MODEL_DIR / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 13. Guardar predicciones de TEST
    # -------------------------------------------------

    output = test_df[
        [
            "rotten_tomatoes_link",
            "review_score",
            "review_type",
            "review_content",
        ]
    ].copy()

    output["true_score"] = true_scores

    output["fresh_rotten_baseline"] = baseline_scores

    output["predicted_residual"] = raw_residuals

    output["predicted_score"] = predicted_scores

    output["absolute_error"] = np.abs(
        true_scores - predicted_scores
    )

    output.to_csv(
        MODEL_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 14. Mostrar resultados
    # -------------------------------------------------

    print(
        "\n=== RESULTADOS TEST — REGRESIÓN RESIDUAL ==="
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== COMPARACIÓN ===")

    print(
        f"Baseline MAE: {mae_baseline:.4f}"
    )

    print(
        f"RoBERTa MAE:  {mae_roberta:.4f}"
    )

    print(
        f"Baseline RMSE: {rmse_baseline:.4f}"
    )

    print(
        f"RoBERTa RMSE:  {rmse_roberta:.4f}"
    )

    improvement = (
        (mae_baseline - mae_roberta)
        / mae_baseline
        * 100
    ) if mae_baseline > 0 else 0.0

    print(
        f"\nMejora relativa MAE: {improvement:.2f}%"
    )

    print(
        f"\nModelo y resultados guardados en: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()