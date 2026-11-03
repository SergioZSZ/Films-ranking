import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# 1. Configuración de la página
st.set_page_config(
    page_title="Movie Ranking",
    page_icon="🎬",
    layout="wide"
)

# 2. Carga de la variable de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# 3. Conexión a la BBDD con caché (para no saturar PostgreSQL)
@st.cache_data(ttl=3600)
def get_data():
    if not DATABASE_URL:
        st.error("⚠️ No se ha encontrado la variable de entorno DATABASE_URL. Revisa tu archivo .env")
        st.stop()
        
    try:
        # Conexión mediante SQLAlchemy
        engine = create_engine(DATABASE_URL)
        
        # Consulta directa a la tabla que creó tu compañero
        query = """
    SELECT 
        r.rank_position AS "Posición",
        r.movie_title AS "Título",
        m.start_year AS "Año",
        r.final_score AS "Nota Final",
        r.imdb_bayesian_rating AS "Nota Bayesiana IMDb",
        r.critic_score AS "Media Críticos",
        (r.observed_review_count + r.predicted_review_count) AS "Número de Críticas"
    FROM movie_ranking r
    LEFT JOIN imdb_movies m ON r.imdb_id = m.tconst
    ORDER BY r.rank_position ASC
"""
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            
        return df
    except Exception as e:
        st.error(f" Error al conectar a la base de datos: {e}")
        st.info(" Consejo: Revisa si los nombres de las columnas (ej: release_year, rt_reviews_count) coinciden exactamente con los de la base de datos.")
        st.stop()

def main():
    st.title("Ranking de Películas")
    st.markdown("Clasificación final combinando el volumen de audiencia (IMDb) con el análisis de sentimiento de los críticos (Rotten Tomatoes).")
    
    # Cargar datos
    df = get_data()
    
    if df.empty:
        st.warning("No hay datos disponibles en la base de datos.")
        return
        
    # --- BARRA LATERAL (Filtros) ---
    st.sidebar.header("Filtros y Búsqueda")
    
    # Buscador por título
    search_term = st.sidebar.text_input("Buscar por título:", "")
    
    # Selector de Top N
    top_n = st.sidebar.selectbox(
        "Mostrar Top:",
        options=[10, 50, 100, 500],
        index=2  # Por defecto muestra el Top 100
    )
    
    # Filtro de Críticas de Rotten Tomatoes
    rt_filter = st.sidebar.radio(
        "Filtro Rotten Tomatoes:",
        options=["Todas", "Con críticas de RT", "Sin críticas de RT"]
    )
    
    # --- APLICAR FILTROS ---
    df_filtered = df.copy()
    
    # Aplicar búsqueda por texto (case insensitive)
    if search_term:
        df_filtered = df_filtered[df_filtered["Título"].str.contains(search_term, case=False, na=False)]
        
    # Aplicar filtro RT 
    if rt_filter == "Con críticas de RT":
        df_filtered = df_filtered[df_filtered["Número de Críticas"].fillna(0) > 0]
    elif rt_filter == "Sin críticas de RT":
        df_filtered = df_filtered[df_filtered["Número de Críticas"].fillna(0) == 0]
        
    # Aplicar límite de TOP
    df_filtered = df_filtered.head(top_n)
    
    # --- VISUALIZACIÓN ---
    # Fila de métricas rápidas
    col1, col2 = st.columns(2)
    col1.metric("Películas mostradas", len(df_filtered))
    if not df_filtered.empty:
        col2.metric("Nota del #1 actual", f"{df_filtered.iloc[0]['Nota Final']:.2f}")
    
    # Mostrar la tabla interactiva
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Posición": st.column_config.NumberColumn("Pos", format="%d"),
            "Título": st.column_config.TextColumn("Título"),
            "Año": st.column_config.NumberColumn("Año", format="%d"),
            "Nota Final": st.column_config.NumberColumn("Nota Final", format="%.2f"),
            "Nota Bayesiana IMDb": st.column_config.NumberColumn("Bayesiana IMDb", format="%.2f"),
            "Media Críticos": st.column_config.NumberColumn("Críticos", format="%.2f"),
            "Número de Críticas": st.column_config.NumberColumn("Nº Críticas", format="%d"),
        }
    )

if __name__ == "__main__":
    main()