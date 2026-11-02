"""Prepara los dos CSV de Rotten Tomatoes para el proyecto EPAC.

Requisitos:
    pip install pandas

Uso:
    1. Configura INPUT_DIR y OUTPUT_DIR.
    2. Ejecuta: python prepare_rotten_tomatoes.py

No modifica los archivos originales. No normaliza puntuaciones ni realiza NLP.


Lee los dos CSV originales de Rotten Tomatoes.
- Comprueba que existen todas las columnas necesarias.
- Elimina las críticas completamente duplicadas.
- Corrige las inconsistencias detectadas entre los identificadores de películas.
- Conserva únicamente las 14 columnas que hemos seleccionado.
- Verifica que todas las críticas están asociadas a una película existente.
- Escala las scores de la critica en una escala 0-10 en normalized_score
- Guarda los dos CSV procesados en el directorio configurado.

"""

from pathlib import Path

import pandas as pd
import re


# =================== CONFIGURACIÓN ===================
INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "original_datasets" / "rottentomatoes"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "processed_datasets" / "rottentomatoes"

MOVIES_FILENAME = "rotten_tomatoes_movies.csv"
REVIEWS_FILENAME = "rotten_tomatoes_critic_reviews.csv"
# =====================================================

MOVIES_COLUMNS = [
    "rotten_tomatoes_link",
    "movie_title",
    "original_release_date",
    "directors",
    "tomatometer_rating",
    "tomatometer_count",
]

REVIEWS_COLUMNS = [
    "rotten_tomatoes_link",
    "critic_name",
    "publisher_name",
    "review_date",
    "top_critic",
    "review_type",
    "review_score",
    "review_content",
]

# Correcciones específicas detectadas al comparar los dos CSV originales.
# Solo modifican identificadores de reviews que no coinciden con movies.
REVIEW_MOVIE_ID_FIXES = {
    "m/-_man": "m/minus_man",
    "m/+h": "m/plush",
    "m/-cule_valley_of_the_lost_ants": "m/minuscule_valley_of_the_lost_ants",
    "m/+_one_2019": "m/plus_one_2019",
    "m/patton_oswalt_tragedy_+_comedy_equals_time": "m/patton_oswalt_tragedy_plus_comedy_equals_time",
    "m/sympathy-for-the-devil-one-+-one": "m/sympathy-for-the-devil-one-plus-one",
}




# =================== NORMALIZACIÓN DE SCORES ===================

LETTER_SCORES = {
    "A+": 10.0,
    "A": 9.5,
    "A-": 9.0,
    "B+": 8.5,
    "B": 8.0,
    "B-": 7.5,
    "C+": 7.0,
    "C": 6.5,
    "C-": 6.0,
    "D+": 5.5,
    "D": 5.0,
    "D-": 4.5,
    "F": 0.0,
}


def normalize_review_score(score: str) -> float | None:
    """
    Normaliza una puntuación de Rotten Tomatoes a una escala 0-10.

    Soporta:
        - Fracciones: 4/5, 3/4, 7/10, 85/100...
        - Calificaciones alfabéticas: A+, A, B-, C...
        - Porcentajes: 85%

    Devuelve:
        float: puntuación normalizada entre 0 y 10.
        None: puntuación ausente, inválida o de escala desconocida.

    Nota:
        La conversión de letras utiliza una escala definida
        para EPAC, no una equivalencia oficial de Rotten Tomatoes.
    """

    if score is None or pd.isna(score):
        return None

    value = re.sub(r"\s+", "", str(score).upper())

    if not value:
        return None

    # 1. Puntuaciones alfabéticas
    if value in LETTER_SCORES:
        return LETTER_SCORES[value]

    # 2. Puntuaciones fraccionarias
    fraction = re.fullmatch(
        r"(\d+(?:[.,]\d+)?)/(\d+(?:[.,]\d+)?)",
        value
    )

    if fraction:
        numerator, denominator = (
            float(x.replace(",", "."))
            for x in fraction.groups()
        )

        if denominator > 0 and 0 <= numerator <= denominator:
            return round(10 * numerator / denominator, 4)

        return None

    # 3. Porcentajes
    percentage = re.fullmatch(
        r"(\d+(?:[.,]\d+)?)%",
        value
    )

    if percentage:
        number = float(
            percentage.group(1).replace(",", ".")
        )

        if 0 <= number <= 100:
            return round(number / 10, 4)

        return None

    # 4. Puntuaciones ambiguas o desconocidas
    return None





def read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo: {path}")

    # Leer todas las columnas inicialmente: así se detectan los duplicados
    # exactos originales ANTES de descartar columnas informativas.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {missing}")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    movies = read_csv(INPUT_DIR / MOVIES_FILENAME, MOVIES_COLUMNS)
    reviews = read_csv(INPUT_DIR / REVIEWS_FILENAME, REVIEWS_COLUMNS)
    original_movies = len(movies)
    original_reviews = len(reviews)

    if movies["rotten_tomatoes_link"].duplicated().any():
        raise ValueError("El archivo de películas tiene IDs duplicados. Revisar antes de continuar.")

    # Quitar duplicados exactos teniendo en cuenta las 8 columnas originales.
    reviews = reviews.drop_duplicates().copy()
    removed_duplicates = original_reviews - len(reviews)

    # Reparar seis IDs de películas incompatibles entre ambos archivos.
    bad_ids = set(reviews["rotten_tomatoes_link"]) - set(movies["rotten_tomatoes_link"])
    if bad_ids - set(REVIEW_MOVIE_ID_FIXES):
        raise ValueError(f"IDs de reviews no identificados en movies: {sorted(bad_ids - set(REVIEW_MOVIE_ID_FIXES))}")

    corrected_rows = reviews["rotten_tomatoes_link"].isin(REVIEW_MOVIE_ID_FIXES).sum()
    reviews["rotten_tomatoes_link"] = reviews["rotten_tomatoes_link"].replace(REVIEW_MOVIE_ID_FIXES)

    # Recortar solo al final: conservar las columnas exactas acordadas.
    movies = movies.loc[:, MOVIES_COLUMNS]
    reviews = reviews.loc[:, REVIEWS_COLUMNS]
    
    # Normalizar puntuaciones de críticas a escala 0-10
    reviews["normalized_score"] = reviews["review_score"].apply(
        normalize_review_score
    )

    # Integridad de la relación películas (1) -> críticas (N).
    orphan_ids = set(reviews["rotten_tomatoes_link"]) - set(movies["rotten_tomatoes_link"])
    if orphan_ids:
        raise ValueError(f"Críticas sin película correspondiente: {sorted(orphan_ids)}")

    movies_output = OUTPUT_DIR / "rotten_tomatoes_movies_clean.csv"
    reviews_output = OUTPUT_DIR / "rotten_tomatoes_critic_reviews_clean.csv"
    movies.to_csv(movies_output, index=False, encoding="utf-8")
    reviews.to_csv(reviews_output, index=False, encoding="utf-8")

    print("=== PREPROCESAMIENTO ROTTEN TOMATOES / EPAC ===")
    print(f"Películas: {original_movies:,} -> {len(movies):,} filas | {len(movies.columns)} columnas")
    print(f"Críticas:  {original_reviews:,} -> {len(reviews):,} filas | {len(reviews.columns)} columnas")
    print(f"Duplicados exactos eliminados: {removed_duplicates:,}")
    print(f"IDs corregidos en filas de críticas: {corrected_rows:,}")
    print(f"Salida películas: {movies_output}")
    print(f"Salida críticas:  {reviews_output}")


if __name__ == "__main__":
    main()
