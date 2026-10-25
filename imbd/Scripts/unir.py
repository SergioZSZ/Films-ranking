from pathlib import Path
import pandas as pd

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / 'Datasets'

# Rutas de los archivos
ratings_file = DATASETS_DIR / 'title.ratings.tsv.gz'
crew_file = DATASETS_DIR / 'title.crew.tsv.gz'
names_file = DATASETS_DIR / 'name.basics.tsv.gz'
basics_file = DATASETS_DIR / 'title.basics.tsv.gz'
output_file = DATASETS_DIR / 'imdb_completo.tsv'

print("1/5 - Cargando ratings...")
df_ratings = pd.read_csv(
    ratings_file,
    sep='\t',
    na_values='\\N',
    compression='gzip'
)

print("2/5 - Cargando y filtrando crew (directores)...")
# Leemos solo tconst y directors
df_crew = pd.read_csv(
    crew_file,
    sep='\t',
    na_values='\\N',
    usecols=['tconst', 'directors'],
    compression='gzip'
)

# Unimos con ratings para quedarnos solo con títulos evaluados
df_ratings_crew = pd.merge(df_ratings, df_crew, on='tconst', how='inner')
del df_ratings
del df_crew

print("3/5 - Identificando directores necesarios y construyendo diccionario de nombres...")
# Obtenemos el conjunto único de IDs de directores que necesitamos mapear
ids_directores_necesarios = set()
for dirs in df_ratings_crew['directors'].dropna():
    for nid in dirs.split(','):
        ids_directores_necesarios.add(nid)

print(f"       -> Se buscarán {len(ids_directores_necesarios):,} directores en name.basics...")

# Leemos name.basics por trozos para no desbordar la RAM, quedándonos solo con los nombres requeridos
dict_nombres = {}
for chunk in pd.read_csv(
    names_file,
    sep='\t',
    na_values='\\N',
    usecols=['nconst', 'primaryName'],
    compression='gzip',
    chunksize=500_000
):
    # Filtramos las personas que sean directores de nuestro conjunto
    chunk_dirs = chunk[chunk['nconst'].isin(ids_directores_necesarios)]
    if not chunk_dirs.empty:
        dict_nombres.update(dict(zip(chunk_dirs['nconst'], chunk_dirs['primaryName'])))

print(f"       -> {len(dict_nombres):,} nombres de directores recuperados con éxito.")

# Función para traducir 'nm0001,nm0002' a 'Nombre 1, Nombre 2'
def mapear_directores(ids):
    if pd.isna(ids):
        return None
    nombres = [dict_nombres.get(nid, nid) for nid in str(ids).split(',')]
    return ', '.join(nombres)

print("       -> Reemplazando IDs por nombres reales en la tabla...")
df_ratings_crew['directors'] = df_ratings_crew['directors'].apply(mapear_directores)

print("4/5 - Procesando y uniendo basics por bloques...")
chunk_size = 500_000
first_chunk = True
total_filas = 0

for chunk in pd.read_csv(
    basics_file,
    sep='\t',
    na_values='\\N',
    compression='gzip',
    low_memory=False,
    chunksize=chunk_size
):
    # Unir bloque de basics con ratings y nombres de directores
    merged_chunk = pd.merge(chunk, df_ratings_crew, on='tconst', how='inner')

    if not merged_chunk.empty:
        merged_chunk.to_csv(
            output_file,
            sep='\t',
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )
        total_filas += len(merged_chunk)
        first_chunk = False

print(f"5/5 - ¡Completado! Se generó '{output_file.name}' con {total_filas:,} títulos y nombres de directores.")