"""Prepara las películas de IMDb y calcula su puntuación bayesiana.

Requisitos:
    pip install pandas

Uso desde la raíz del proyecto:
    python src/imdb/imdb_processment.py

Lee title.basics.tsv.gz, title.ratings.tsv.gz, title.crew.tsv.gz y
name.basics.tsv.gz de original_datasets/imdb sin modificar los originales.
Guarda imdb_peliculas.tsv e imdb_peliculas_bayesiano.tsv en
processed_datasets/imdb. No genera el archivo intermedio imdb_completo.tsv.
"""

from pathlib import Path

import pandas as pd
from pandas.io.parsers import TextFileReader


# =================== CONFIGURACIÓN ===================
INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "original_datasets" / "imdb"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "processed_datasets" / "imdb"

BASICS_FILENAME = "title.basics.tsv.gz"
RATINGS_FILENAME = "title.ratings.tsv.gz"
CREW_FILENAME = "title.crew.tsv.gz"
NAMES_FILENAME = "name.basics.tsv.gz"

CHUNK_SIZE = 500_000
# =====================================================

BASICS_COLUMNS = [
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "runtimeMinutes",
    "genres",
]
RATINGS_COLUMNS = ["tconst", "averageRating", "numVotes"]
CREW_COLUMNS = ["tconst", "directors"]
NAMES_COLUMNS = ["nconst", "primaryName"]
MOVIES_COLUMNS = [column for column in BASICS_COLUMNS if column != "titleType"] + [
    "averageRating", "numVotes", "directors"
]


def read_tsv(
    path: Path, required_columns: list[str], chunksize: int | None = None
) -> pd.DataFrame | TextFileReader:
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo: {path}")

    columns = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8").columns
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {missing}")

    return pd.read_csv(
        path,
        sep="\t",
        na_values="\\N",
        usecols=required_columns,
        low_memory=False,
        chunksize=chunksize,
        encoding="utf-8",
    )


def load_movies() -> pd.DataFrame:
    print("1/4 - Cargando puntuaciones y filtrando películas por bloques...")
    ratings = read_tsv(INPUT_DIR / RATINGS_FILENAME, RATINGS_COLUMNS)
    movie_chunks = []
    with read_tsv(INPUT_DIR / BASICS_FILENAME, BASICS_COLUMNS, CHUNK_SIZE) as reader:
        for chunk in reader:
            movies = chunk.loc[chunk["titleType"] == "movie"].drop(columns="titleType")
            merged = movies.merge(ratings, on="tconst", how="inner")
            if not merged.empty:
                movie_chunks.append(merged)

    if not movie_chunks:
        raise ValueError("No se encontraron películas con puntuación en los archivos de IMDb.")
    return pd.concat(movie_chunks, ignore_index=True)


def add_directors(movies: pd.DataFrame) -> pd.DataFrame:
    print("2/4 - Incorporando directores por bloques...")
    movie_ids = set(movies["tconst"])
    crew_chunks = []
    with read_tsv(INPUT_DIR / CREW_FILENAME, CREW_COLUMNS, CHUNK_SIZE) as reader:
        for chunk in reader:
            crew = chunk.loc[chunk["tconst"].isin(movie_ids)]
            if not crew.empty:
                crew_chunks.append(crew)

    if not crew_chunks:
        raise ValueError("No se encontraron películas con datos de equipo en los archivos de IMDb.")
    movies = movies.merge(pd.concat(crew_chunks, ignore_index=True), on="tconst", how="inner")

    director_ids = {
        director_id
        for directors in movies["directors"].dropna()
        for director_id in directors.split(",")
    }
    director_names = {}
    with read_tsv(INPUT_DIR / NAMES_FILENAME, NAMES_COLUMNS, CHUNK_SIZE) as reader:
        for chunk in reader:
            directors = chunk.loc[chunk["nconst"].isin(director_ids)]
            director_names.update(zip(directors["nconst"], directors["primaryName"]))

    def map_directors(ids: str) -> str | None:
        if pd.isna(ids):
            return None
        return ", ".join(director_names.get(director_id, director_id) for director_id in ids.split(","))

    movies["directors"] = movies["directors"].apply(map_directors)
    return movies.loc[:, MOVIES_COLUMNS]


def main() -> None:
    for filename in (BASICS_FILENAME, RATINGS_FILENAME, CREW_FILENAME, NAMES_FILENAME):
        path = INPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra el archivo: {path}")

    movies = add_directors(load_movies())
    if movies["tconst"].duplicated().any():
        raise ValueError("El archivo de películas tiene IDs duplicados. Revisar antes de continuar.")

    print("3/4 - Calculando puntuaciones bayesianas...")
    # Media por película, sin ponderar por votos, igual que en Escalado.py.
    global_mean = movies["averageRating"].mean()
    # El percentil 90 regula el ajuste; no se descartan películas con menos votos.
    vote_threshold = movies["numVotes"].quantile(0.90)
    votes = movies["numVotes"]
    ratings = movies["averageRating"]
    bayesian_rating = (
        (votes / (votes + vote_threshold)) * ratings
        + (vote_threshold / (votes + vote_threshold)) * global_mean
    )
    ranking = movies.assign(bayesianRating=bayesian_rating.round(2))
    ranking = ranking.sort_values(by="bayesianRating", ascending=False)

    print("4/4 - Guardando datasets procesados...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movies_output = OUTPUT_DIR / "imdb_peliculas.tsv"
    ranking_output = OUTPUT_DIR / "imdb_peliculas_bayesiano.tsv"
    movies.to_csv(movies_output, sep="\t", index=False, encoding="utf-8")
    ranking.to_csv(ranking_output, sep="\t", index=False, encoding="utf-8")

    print("=== PREPROCESAMIENTO IMDB / EPAC ===")
    print(f"Películas: {len(movies):,} filas | {len(movies.columns)} columnas")
    print(f"Media global (C): {global_mean:.4f}")
    print(f"Percentil 90 de votos (m): {vote_threshold:g}")
    print(f"Salida películas: {movies_output}")
    print(f"Salida ranking:   {ranking_output}")


if __name__ == "__main__":
    main()
