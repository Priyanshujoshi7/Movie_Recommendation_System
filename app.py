import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------
# PAGE SETTINGS
# -------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("archive/movies.csv")
    ratings = pd.read_csv("archive/ratings.csv")
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    return movies, ratings

movies, ratings = load_data()

# Map ids → titles
movie_titles = movies.set_index("movieId")["title"].to_dict()

# -------------------------------------------
# SIMILARITY MATRIX
# -------------------------------------------
@st.cache_resource
def build_similarity(ratings_df):
    user_item_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    item_user_matrix = user_item_matrix.T

    sim = cosine_similarity(item_user_matrix)
    return pd.DataFrame(sim, index=item_user_matrix.index, columns=item_user_matrix.index)

similarity_df = build_similarity(ratings)

# -------------------------------------------
# SESSION STATE
# -------------------------------------------
if "liked_movies" not in st.session_state:
    st.session_state.liked_movies = []

if "excluded_movies" not in st.session_state:
    st.session_state.excluded_movies = []

if "last_recs" not in st.session_state:
    st.session_state.last_recs = None


# -------------------------------------------
# RECOMMENDER FUNCTION
# -------------------------------------------
def recommend_movies(user_likes, selected_genre, excluded_ids, top_n=10):
    all_scores = {}

    for movie_id in user_likes:
        if movie_id not in similarity_df.index:
            continue

        for sim_movie_id, score in similarity_df[movie_id].items():
            if sim_movie_id in user_likes or sim_movie_id in excluded_ids:
                continue
            all_scores[sim_movie_id] = all_scores.get(sim_movie_id, 0) + score

    # Sort by score
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [m for m, _ in ranked[:150]]

    candidates = movies[movies["movieId"].isin(top_ids)].copy()

    if selected_genre != "All":
        candidates = candidates[candidates["genres"].apply(lambda gs: selected_genre in gs)]

    return candidates.head(top_n)


# -------------------------------------------
# SIDEBAR — LIKED MOVIES
# -------------------------------------------
st.sidebar.header("❤️ Movies You Like")

if st.session_state.liked_movies:
    for mid in st.session_state.liked_movies:
        st.sidebar.write(f"• {movie_titles[mid]}")
else:
    st.sidebar.write("No liked movies yet.")


# -------------------------------------------
# UI — MAIN LAYOUT
# -------------------------------------------
st.title("🎬 Movie Recommendation System")

col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("1️⃣ Select Movies You Like")

    selected_likes = st.multiselect(
        "🎞️ Choose one or more movies:",
        options=movies["movieId"].tolist(),
        format_func=lambda x: movie_titles[x]
    )

    st.subheader("2️⃣ Select Genre")
    all_genres = sorted({g for row in movies["genres"] for g in row})
    selected_genre = st.selectbox("Genre:", ["All"] + all_genres)

    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.liked_movies = []
        st.session_state.excluded_movies = []
        st.session_state.last_recs = None
        st.rerun()

    if st.button("✨ Get Recommendations", use_container_width=True):
        st.session_state.last_recs = recommend_movies(
            user_likes=selected_likes,
            selected_genre=selected_genre,
            excluded_ids=st.session_state.excluded_movies,
            top_n=10
        )


# -------------------------------------------
# OUTPUT — SHOW RECOMMENDED MOVIES
# -------------------------------------------
with col_output:
    st.subheader("⭐ Recommended For You")

    recs = st.session_state.last_recs

    if recs is None:
        st.info("Select movies and click *Get Recommendations*.")
    elif recs.empty:
        st.warning("No recommendations found. Try changing genre.")
    else:
        for _, row in recs.iterrows():
            movie_id = int(row["movieId"])
            title = row["title"]
            genres_text = ", ".join(row["genres"])

            card = st.container()
            with card:
                c1, c2, c3 = st.columns([4, 2, 2])

                with c1:
                    st.markdown(f"**🎬 {title}**")
                    st.caption(genres_text)

                with c2:
                    if st.button("❤️ Like", key=f"like_{movie_id}"):
                        if movie_id not in st.session_state.liked_movies:
                            st.session_state.liked_movies.append(movie_id)
                        st.rerun()

                with c3:
                    if st.button("👀 Remove", key=f"remove_{movie_id}"):
                        st.session_state.excluded_movies.append(movie_id)

                        # Recompute fresh recommendations  
                        new_recs = recommend_movies(
                            user_likes=selected_likes,
                            selected_genre=selected_genre,
                            excluded_ids=st.session_state.excluded_movies,
                            top_n=10
                        )
                        st.session_state.last_recs = new_recs

                        # Simulated "animation": remove → replace instantly
                        st.rerun()

            st.markdown("---")
