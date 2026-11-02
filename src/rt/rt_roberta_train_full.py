"""Entrenamiento FINAL de RoBERTa residual con todas las críticas etiquetadas.

Ejecutar desde la raíz de Films-ranking:
    poetry run python src/rt/train_roberta_residual_full.py

No realiza predicciones ni separa train/validation/test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_CSV = Path(
    "processed_datasets/rottentomatoes/rotten_tomatoes_critic_reviews_clean.csv"
)
DEFAULT_OUT = Path("models/epac-roberta-residual-full")
MODEL_NAME = "FacebookAI/roberta-base"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true",
                        help="Continuar desde el último checkpoint del directorio de salida")
    parser.add_argument("--fp16", action="store_true",
                        help="Usar precisión mixta si hay GPU CUDA")
    return parser.parse_args()


def load_labeled_reviews(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    text_col = next(
        (c for c in ("review_content", "review_text") if c in df.columns), None
    )
    if text_col is None or not {"review_type", "normalized_score"}.issubset(df.columns):
        raise ValueError(
            "El CSV necesita review_type, normalized_score y "
            "review_content (o review_text)."
        )

    total = len(df)

    # Nunca entrenar con críticas que carecen de nota numérica original.
    if "review_score" in df.columns:
        original_score = df["review_score"].astype("string").str.strip()
        df = df.loc[original_score.notna() & original_score.ne("")].copy()
    else:
        print("AVISO: no existe review_score; se presupone que normalized_score "
              "contiene exclusivamente notas reales, NO predicciones.")

    df["normalized_score"] = pd.to_numeric(df["normalized_score"], errors="coerce")
    df["review_type"] = df["review_type"].astype("string").str.strip().str.capitalize()
    df["text"] = df[text_col].astype("string").str.strip()

    df = df.loc[
        df["normalized_score"].between(0, 10)
        & df["review_type"].isin(["Fresh", "Rotten"])
        & df["text"].notna()
        & df["text"].ne("")
    ].copy()

    if df.empty or set(df["review_type"].unique()) != {"Fresh", "Rotten"}:
        raise ValueError("Se necesitan críticas válidas tanto Fresh como Rotten.")

    # Baselines calculados sobre TODO el corpus etiquetado definitivo.
    means = {
        key: float(val)
        for key, val in df.groupby("review_type")["normalized_score"].mean().items()
    }
    df["labels"] = (
        df["normalized_score"] - df["review_type"].map(means)
    ).astype("float32")

    texts = (
        "Sentiment classification: " + df["review_type"].astype(str)
        + ". Review: " + df["text"].astype(str)
    ).tolist()

    counts = {k: int(v) for k, v in df["review_type"].value_counts().items()}
    print(f"Filas originales: {total:,}")
    print(f"Críticas con puntuación numérica UTILIZABLE: {len(df):,}")
    print(f"Fresh: {counts.get('Fresh', 0):,} | Rotten: {counts.get('Rotten', 0):,}")
    print(f"Medias recalculadas: {means}")
    print(f"Filas descartadas (sin nota/texto válido): {total - len(df):,}")

    dataset = Dataset.from_dict({"text": texts, "labels": df["labels"].tolist()})
    dataset = dataset.cast_column("labels", Value("float32"))
    return dataset, means, counts


def main():
    args = parse_args()
    if args.fp16 and not torch.cuda.is_available():
        raise ValueError("--fp16 requiere una GPU disponible mediante PyTorch CUDA")
    if args.epochs <= 0 or args.batch_size < 1 or args.save_steps < 1:
        raise ValueError("epochs, batch-size y save-steps deben ser positivos")

    set_seed(42)
    output = args.output
    last_checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
    if args.resume:
        if last_checkpoint is None:
            raise FileNotFoundError("No hay checkpoint para reanudar en " + str(output))
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"La carpeta {output} ya tiene archivos. Usa --resume para "
            "continuar o elige otra --output para no sobrescribir modelos."
        )

    dataset, means, counts = load_labeled_reviews(args.csv)
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "training_metadata.json"
    metadata = {
        "model": MODEL_NAME,
        "objective": "normalized_score - mean(review_type)",
        "input_template": "Sentiment classification: {review_type}. Review: {review_content}",
        "postprocessing": "clip(mean(review_type) + predicted_residual, 0, 10)",
        "csv_source": str(args.csv.resolve()),
        "train_examples": len(dataset),
        "counts": counts,
        "means": means,
        "seed": 42,
        "max_length": 256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "no_validation_or_prediction": True,
    }
    if args.resume:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for field in ("csv_source", "train_examples", "counts", "means", "epochs", "batch_size"):
            if previous[field] != metadata[field]:
                raise ValueError(f"Los datos/configuración cambiaron: {field}; no reanudes este checkpoint")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando críticas",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression"
    )
    train_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        fp16=args.fp16,
        seed=42,
        data_seed=42,
        logging_steps=100,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,  # compatible con Windows
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print(f"\nDispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Entrenando {len(dataset):,} ejemplos durante {args.epochs} épocas...")
    metrics = trainer.train(resume_from_checkpoint=last_checkpoint).metrics
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    metadata["training_metrics"] = {
        k: float(v) if isinstance(v, (float, np.floating)) else v
        for k, v in metrics.items()
    }
    metadata["status"] = "completed"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModelo final: {output.resolve()}")
    print(f"Metadatos y medias: {meta_path.resolve()}")
    print("Finalizado. No se han predicho críticas ni calculado métricas de test.")


if __name__ == "__main__":
    main()
