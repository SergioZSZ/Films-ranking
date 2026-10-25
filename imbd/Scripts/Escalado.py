from pathlib import Path
import pandas as pd

# Directorio raíz del proyecto (imbd/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / 'Datasets'

input_file = DATASETS_DIR / 'imdb_peliculas.tsv'
output_file = DATASETS_DIR / 'imdb_peliculas_bayesiano.tsv'

print(f"Cargando {input_file.name}...")
df = pd.read_csv(input_file, sep='\t', low_memory=False)

# C = Calificación media de todas las películas
C = df['averageRating'].mean()

# m = Umbral mínimo de votos (percentil 90)
m = df['numVotes'].quantile(0.90)

print(f"Media global (C): {C:.2f}")
print(f"Votos mínimos percentil 90 (m): {m:.0f}")

# Cálculo vectorial de la fórmula bayesiana (mucho más rápido que apply fila por fila)
v = df['numVotes']
R = df['averageRating']
df['bayesianRating'] = ((v / (v + m)) * R) + ((m / (v + m)) * C)

# Redondeamos a dos decimales y ordenamos de mayor a menor puntuación bayesiana
df['bayesianRating'] = df['bayesianRating'].round(2)
df = df.sort_values(by='bayesianRating', ascending=False)

print(f"Guardando resultado en {output_file.name}...")
df.to_csv(output_file, sep='\t', index=False)

print(f"¡Completado! Archivo guardado con {len(df):,} películas ordenadas por 'bayesianRating'.")