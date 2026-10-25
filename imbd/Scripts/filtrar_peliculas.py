from pathlib import Path
import pandas as pd

# Directorio raíz del proyecto (imbd/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas hacia la carpeta Datasets
input_file = BASE_DIR / 'Datasets' / 'imdb_completo.tsv'
output_file = BASE_DIR / 'Datasets' / 'imdb_peliculas.tsv'

print("Filtrando películas y eliminando columnas 'endYear' y 'titleType'...")

chunk_size = 500_000
first_chunk = True
total_peliculas = 0

# Columnas a descartar del archivo final
columnas_a_borrar = ['endYear', 'titleType']

for chunk in pd.read_csv(input_file, sep='\t', low_memory=False, chunksize=chunk_size):
    # 1. Filtramos para conservar únicamente 'movie'
    movies_chunk = chunk[chunk['titleType'] == 'movie'].copy()
    
    if not movies_chunk.empty:
        # 2. Eliminamos las columnas tras aplicar el filtro
        movies_chunk.drop(columns=columnas_a_borrar, inplace=True, errors='ignore')
        
        # 3. Guardamos el resultado en el archivo final
        movies_chunk.to_csv(
            output_file,
            sep='\t',
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )
        total_peliculas += len(movies_chunk)
        first_chunk = False

print(f"¡Listo! Se guardaron {total_peliculas:,} películas en '{output_file.name}' sin 'titleType' ni 'endYear'.")