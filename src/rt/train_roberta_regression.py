"""EPAC: fine-tuning de RoBERTa-base para inferir notas 0-10 de críticas.

Entrena SOLO con notas explícitas fraccionarias (por ejemplo 3/4, 4/5).
La separación train/valid/test se hace POR PELÍCULA para evitar fugas.

Requisitos (en un entorno Python con PyTorch):
    pip install torch transformers datasets accelerate scikit-learn pandas numpy

Antes de ejecutar, ajusta INPUT_CSV, MODEL_DIR y los máximos por partición.
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
    Trainer,
    TrainingArguments,
)

# =========================== CONFIGURACIÓN ===========================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent 
INPUT_CSV = PROJECT_DIR / "processed_datasets" / "rottentomatoes" / "rotten_tomatoes_critic_reviews_clean.csv"
MODEL_DIR = PROJECT_DIR / "models" / "epac-roberta-regression"
BASE_MODEL = "FacebookAI/roberta-base"
SEED = 42
MAX_TOKENS = 256
MAX_TRAIN = 50_000  # Primer experimento; incrementar tras validar.
MAX_VALID = 5_000
MAX_TEST = 5_000
EPOCHS = 2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
# =====================================================================

FRACTION_PATTERN = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)\s*$")


def load_reliable_examples(path: Path) -> pd.DataFrame:
    """Usa únicamente reseñas con nota de denominador explícito y texto."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el CSV: {path}")

    cols = ["rotten_tomatoes_link", "review_score", "review_type", "review_content"]
    frame = pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False)
    frame["review_content"] = frame["review_content"].str.strip()
    frame["review_score"] = frame["review_score"].str.strip()

    matches = frame["review_score"].str.match(FRACTION_PATTERN)
    frame = frame.loc[matches & frame["review_content"].ne("")].copy()

    parts = frame["review_score"].str.extract(FRACTION_PATTERN.pattern)
    numerator = pd.to_numeric(parts[0].str.replace(",", ".", regex=False), errors="coerce")
    denominator = pd.to_numeric(parts[1].str.replace(",", ".", regex=False), errors="coerce")
    valid = denominator.gt(0) & numerator.ge(0) & numerator.le(denominator)
    frame = frame.loc[valid].copy()
    frame["target"] = (10 * numerator.loc[valid] / denominator.loc[valid]).astype("float32")

    # Evitar que textos exactamente idénticos aparezcan en distintas particiones
    # mediante una exclusión posterior de solapamientos.
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No hay críticas válidas con texto y puntuación fraccionaria.")
    return frame


def split_by_movie(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición reproducible por IDs de película (no por fila)."""
    groups = frame["rotten_tomatoes_link"]
    split_outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_val_idx, test_idx = next(split_outer.split(frame, groups=groups))

    train_val_frame = frame.iloc[train_val_idx]
    split_inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=SEED + 1)
    train_rel, valid_rel = next(
        split_inner.split(train_val_frame, groups=train_val_frame["rotten_tomatoes_link"])
    )

    train = frame.iloc[train_val_idx[train_rel]].copy()
    valid = frame.iloc[train_val_idx[valid_rel]].copy()
    test = frame.iloc[test_idx].copy()

    # Si el mismo texto aparece en títulos distintos, no debe pasar de train a valid/test.
    heldout_texts = set(valid["review_content"]) | set(test["review_content"])
    train = train.loc[~train["review_content"].isin(heldout_texts)].copy()
    test = test.loc[~test["review_content"].isin(set(valid["review_content"]))].copy()

    for name, subset in (("train", train), ("validation", valid), ("test", test)):
        if subset.empty:
            raise ValueError(f"Partición {name} vacía; comprobar los datos.")
    assert set(train.rotten_tomatoes_link).isdisjoint(valid.rotten_tomatoes_link)
    assert set(train.rotten_tomatoes_link).isdisjoint(test.rotten_tomatoes_link)
    assert set(valid.rotten_tomatoes_link).isdisjoint(test.rotten_tomatoes_link)

    def sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(df) > limit:
            df = df.sample(n=limit, random_state=SEED)
        return df.reset_index(drop=True)

    return sample(train, MAX_TRAIN), sample(valid, MAX_VALID), sample(test, MAX_TEST)


def as_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    data = pd.DataFrame({
        "review_content": frame["review_content"],
        "labels": frame["target"].astype("float32"),
    })
    dataset = Dataset.from_pandas(data, preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["review_content"], truncation=True, max_length=MAX_TOKENS)

    return dataset.map(tokenize, batched=True, remove_columns=["review_content"])


def compute_metrics(prediction):
    y_pred = np.clip(np.asarray(prediction.predictions).reshape(-1), 0, 10)
    y_true = np.asarray(prediction.label_ids).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: no se detecta GPU CUDA. El entrenamiento en CPU puede ser muy lento.")

    data = load_reliable_examples(INPUT_CSV)
    train_df, valid_df, test_df = split_by_movie(data)
    print(f"Ejemplos con escala explícita y texto: {len(data):,}")
    print(f"Train/valid/test: {len(train_df):,} / {len(valid_df):,} / {len(test_df):,}")
    print("La separación se ha realizado por película, antes de muestrear.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    ds_train = as_dataset(train_df, tokenizer)
    ds_valid = as_dataset(valid_df, tokenizer)
    ds_test = as_dataset(test_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, problem_type="regression"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    raw_predictions = trainer.predict(ds_test).predictions.reshape(-1)
    predictions = np.clip(raw_predictions, 0.0, 10.0)
    target = test_df["target"].to_numpy(dtype=float)

    # Control sencillo: puntuación media del train por Fresh/Rotten.
    # Para este baseline se utiliza review_type del test, pero no para RoBERTa.
    means = train_df.groupby("review_type")["target"].mean().to_dict()
    fallback = float(train_df["target"].mean())
    simple_baseline = test_df["review_type"].map(means).fillna(fallback).to_numpy(dtype=float)

    metrics = {
        "model": BASE_MODEL,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "train_rows": len(train_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_movies": int(train_df.rotten_tomatoes_link.nunique()),
        "validation_movies": int(valid_df.rotten_tomatoes_link.nunique()),
        "test_movies": int(test_df.rotten_tomatoes_link.nunique()),
        "test_mae_roberta": float(mean_absolute_error(target, predictions)),
        "test_rmse_roberta": float(np.sqrt(mean_squared_error(target, predictions))),
        "test_mae_fresh_rotten_baseline": float(mean_absolute_error(target, simple_baseline)),
        "test_rmse_fresh_rotten_baseline": float(np.sqrt(mean_squared_error(target, simple_baseline))),
        "train_group_means": {k: float(v) for k, v in means.items()},
        "training_labels": "only numeric fractions with explicit denominator",
        "source_csv": str(INPUT_CSV),
    }

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    out = test_df[["rotten_tomatoes_link", "review_score", "review_type", "review_content"]].copy()
    out["true_score"] = target
    out["predicted_score"] = predictions
    out["fresh_rotten_baseline"] = simple_baseline
    out.to_csv(MODEL_DIR / "test_predictions.csv", index=False, encoding="utf-8")

    print("\n=== RESULTADOS TEST (películas no vistas durante entrenamiento) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModelo y métricas guardados en: {MODEL_DIR}")


if __name__ == "__main__":
    main()
