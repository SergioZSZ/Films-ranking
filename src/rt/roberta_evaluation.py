"""EPAC: evaluar predicciones residuales agregadas por película.
Colocar en src/rt/rt_evaluate_aggregation.py y ejecutar desde la raíz del repo:
    poetry run python src/rt/rt_evaluate_aggregation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / 'models' / 'epac-roberta-residual'
INPUT_CSV = MODEL_DIR / 'test_predictions.csv'
OUTPUT_DIR = MODEL_DIR / 'aggregation_evaluation'
BOOTSTRAP_REPEATS = 2000
SEED = 42

REQUIRED = ['rotten_tomatoes_link', 'true_score', 'predicted_score', 'fresh_rotten_baseline']


def metricas(datos: pd.DataFrame) -> dict:
    real = datos['true_score'].to_numpy(float)
    resultado = {}
    for nombre, campo in [('roberta', 'predicted_score'), ('baseline', 'fresh_rotten_baseline')]:
        err = datos[campo].to_numpy(float) - real
        resultado[nombre] = {
            'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt(np.square(err).mean())),
            'sesgo': float(err.mean()),
        }
    resultado['mejora_mae'] = resultado['baseline']['mae'] - resultado['roberta']['mae']
    resultado['n'] = int(len(datos))
    return resultado


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f'No se encuentra: {INPUT_CSV}. Coloca este script en src/rt/.')
    criticas = pd.read_csv(INPUT_CSV)
    faltan = set(REQUIRED) - set(criticas.columns)
    if faltan:
        raise ValueError(f'Faltan columnas: {sorted(faltan)}')
    criticas = criticas[REQUIRED].copy()
    if criticas.empty:
        raise ValueError('test_predictions.csv está vacío')
    if criticas['rotten_tomatoes_link'].isna().any():
        raise ValueError('Hay películas sin identificador')
    for col in REQUIRED[1:]:
        criticas[col] = pd.to_numeric(criticas[col], errors='raise')
    if not np.isfinite(criticas[REQUIRED[1:]].to_numpy(float)).all():
        raise ValueError('Hay puntuaciones ausentes o no finitas')

    # Misma importancia para cada película, independientemente del nº de reseñas.
    peliculas = criticas.groupby('rotten_tomatoes_link', as_index=False).agg(
        n_criticas=('true_score', 'size'),
        true_score=('true_score', 'mean'),
        predicted_score=('predicted_score', 'mean'),
        fresh_rotten_baseline=('fresh_rotten_baseline', 'mean'),
    )
    peliculas['error_roberta'] = (peliculas['predicted_score'] - peliculas['true_score']).abs()
    peliculas['error_baseline'] = (peliculas['fresh_rotten_baseline'] - peliculas['true_score']).abs()
    peliculas['mejora_error'] = peliculas['error_baseline'] - peliculas['error_roberta']

    # Bootstrap pareado por película, no por reseña.
    deltas = peliculas['mejora_error'].to_numpy(float)
    rng = np.random.default_rng(SEED)
    muestras = np.array([
        deltas[rng.integers(0, len(deltas), len(deltas))].mean()
        for _ in range(BOOTSTRAP_REPEATS)
    ])
    ic95 = np.quantile(muestras, [0.025, 0.975]).tolist()

    por_minimo = {}
    for minimo in (1, 3, 5, 10):
        subset = peliculas[peliculas['n_criticas'] >= minimo]
        por_minimo[str(minimo)] = metricas(subset) if len(subset) else None

    resumen = {
        'por_critica': metricas(criticas),
        'por_pelicula': metricas(peliculas),
        'ic95_mejora_mae_pelicula': ic95,
        'bootstrap_repeticiones': BOOTSTRAP_REPEATS,
        'por_pelicula_minimo_criticas_test': por_minimo,
        'advertencia': 'Las medias reales se basan solo en las críticas muestreadas de test; no son medias completas de las películas.'
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    peliculas.to_csv(OUTPUT_DIR / 'per_movie.csv', index=False)
    peliculas.nlargest(30, 'error_roberta').to_csv(OUTPUT_DIR / 'worst_30_movies.csv', index=False)
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== POR CRÍTICA (comprobación) ===')
    print(json.dumps(resumen['por_critica'], indent=2, ensure_ascii=False))
    print('\n=== POR PELÍCULA ===')
    print('Películas:', len(peliculas), ' / mediana críticas por película:', peliculas.n_criticas.median())
    print(json.dumps(resumen['por_pelicula'], indent=2, ensure_ascii=False))
    print('IC 95% bootstrap de la mejora MAE (baseline menos RoBERTa):', ic95)
    print('\n=== PELÍCULAS CON AL MENOS N CRÍTICAS EN TEST ===')
    print(json.dumps(por_minimo, indent=2, ensure_ascii=False))
    print('\nResultados en:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
