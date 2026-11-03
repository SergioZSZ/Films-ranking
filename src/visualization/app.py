"""Explore the combined, Bayesian IMDb and professional critics rankings."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


st.set_page_config(page_title="Movie Ranking", page_icon="🎬", layout="wide")

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
PREDICTION_WEIGHT = 0.8

RANKING_QUERIES = {
    "Final Ranking": """
        SELECT
            r.rank_position AS "Rank",
            r.movie_title AS "Title",
            m.start_year AS "Year",
            r.final_score AS "Final Score",
            r.imdb_bayesian_rating AS "IMDb Bayesian Rating",
            r.critic_score AS "Critic Average",
            (r.observed_review_count + r.predicted_review_count) AS "Review Count"
        FROM movie_ranking r
        LEFT JOIN imdb_movies m ON r.imdb_id = m.tconst
        ORDER BY r.rank_position ASC
    """,
    "IMDb Ranking": """
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY bayesian_rating DESC, num_votes DESC, tconst ASC
            ) AS "Rank",
            primary_title AS "Title",
            start_year AS "Year",
            bayesian_rating AS "IMDb Bayesian Rating",
            average_rating AS "IMDb User Rating",
            num_votes AS "Vote Count"
        FROM imdb_movies
        ORDER BY "Rank" ASC
    """,
    "Critics Ranking": """
        WITH eligible_reviews AS (
            SELECT rotten_tomatoes_link, final_score,
                CASE WHEN score_source = 'observed' THEN 1.0
                     ELSE :prediction_weight END AS weight
            FROM rt_reviews
            WHERE final_score IS NOT NULL
              AND score_source IN ('observed', 'predicted')
        ), critic_scores AS (
            SELECT rotten_tomatoes_link,
                SUM(final_score * weight) / NULLIF(SUM(weight), 0) AS critic_score,
                COUNT(*) AS review_count,
                SUM(weight) AS effective_count
            FROM eligible_reviews
            GROUP BY rotten_tomatoes_link
        )
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY c.critic_score DESC, c.effective_count DESC, m.rt_id ASC
            ) AS "Rank",
            m.movie_title AS "Title",
            m.original_release_date AS "Release Date",
            c.critic_score AS "Critic Average",
            c.review_count AS "Review Count",
            c.effective_count AS "Effective Review Count"
        FROM critic_scores c
        JOIN rt_movies m ON m.rt_id = c.rotten_tomatoes_link
        WHERE c.critic_score IS NOT NULL
        ORDER BY "Rank" ASC
    """,
}


@st.cache_data(ttl=3600)
def get_data(ranking_mode: str) -> pd.DataFrame:
    """Cache each ranking independently, preserving its global positions."""
    if not DATABASE_URL:
        st.error("DATABASE_URL is missing. Check your environment or .env file.")
        st.stop()

    engine = None
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            return pd.read_sql(
                text(RANKING_QUERIES[ranking_mode]),
                connection,
                params={"prediction_weight": PREDICTION_WEIGHT},
            )
    except Exception:
        st.error("Could not load the ranking from the database.")
        st.info("Check the database connection, migrations and data loaders.")
        st.stop()
    finally:
        if engine is not None:
            engine.dispose()


def main() -> None:
    st.sidebar.header("Ranking and Search")
    ranking_mode = st.sidebar.radio("Ranking", options=list(RANKING_QUERIES))
    search_term = st.sidebar.text_input("Search by title", "")
    top_n = st.sidebar.selectbox("Show top", options=[10, 50, 100, 500], index=2)

    st.title(ranking_mode)
    if ranking_mode == "Final Ranking":
        st.markdown(
            "IMDb Bayesian ratings adjusted by professional Rotten Tomatoes reviews, "
            "including scores estimated by RoBERTa."
        )
        score_column = "Final Score"
    elif ranking_mode == "IMDb Ranking":
        st.markdown(
            "Movies ranked by IMDb Bayesian rating, without the Rotten Tomatoes "
            "adjustment. Ties are resolved by vote count and IMDb ID."
        )
        score_column = "IMDb Bayesian Rating"
    else:
        st.markdown(
            "Movies ranked by their weighted critic average (0–10). Observed "
            f"scores weigh 1 and RoBERTa predictions weigh {PREDICTION_WEIGHT:g}. "
            "Includes Rotten Tomatoes movies without an IMDb match. "
            "Ties are resolved by effective review count and Rotten Tomatoes ID."
        )
        score_column = "Critic Average"

    data = get_data(ranking_mode)
    if data.empty:
        st.warning("No movies are available for this ranking.")
        return

    if search_term:
        data = data[
            data["Title"].str.contains(search_term, case=False, na=False, regex=False)
        ]
    displayed = data.head(top_n)

    if displayed.empty:
        st.info("No movies match your search.")
        return

    st.caption("Rank positions refer to the complete selected ranking.")
    count_metric, score_metric = st.columns(2)
    count_metric.metric("Movies shown", len(displayed))
    score_metric.metric("Top displayed score", f"{displayed.iloc[0][score_column]:.2f}")

    st.dataframe(
        displayed,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Title": st.column_config.TextColumn("Title"),
            "Year": st.column_config.NumberColumn("Year", format="%d"),
            "Release Date": st.column_config.DateColumn(
                "Release Date", format="YYYY-MM-DD"
            ),
            "Final Score": st.column_config.NumberColumn("Final Score", format="%.2f"),
            "IMDb Bayesian Rating": st.column_config.NumberColumn(
                "IMDb Bayesian Rating", format="%.2f"
            ),
            "IMDb User Rating": st.column_config.NumberColumn(
                "IMDb User Rating", format="%.2f"
            ),
            "Critic Average": st.column_config.NumberColumn("Critic Average", format="%.2f"),
            "Review Count": st.column_config.NumberColumn("Review Count", format="%d"),
            "Effective Review Count": st.column_config.NumberColumn(
                "Effective Review Count", format="%.1f"
            ),
            "Vote Count": st.column_config.NumberColumn("Vote Count", format="%d"),
        },
    )


if __name__ == "__main__":
    main()
