import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------

# We load the MovieLens dataset from your local "archive" folder.
# movies.csv contains movie titles & genres
# ratings.csv contains user ratings for each movie
movies = pd.read_csv("archive/movies.csv")
ratings = pd.read_csv("archive/ratings.csv")

# Convert the "genres" column from a single string
# into a list of genres (e.g., "Action|Comedy" → ["Action","Comedy"])
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# -----------------------------------------------------------
# STEP 2: CREATE USER–ITEM MATRIX
# -----------------------------------------------------------

# Convert the ratings into a "pivot table" where:
# Rows   = userId
# Columns = movieId
# Values  = the rating given by the user
# Missing values are filled with 0 (meaning: user did not rate)
user_item_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# For cosine similarity, we need movies × users instead of users × movies
item_user_matrix = user_item_matrix.T  # transpose the matrix

# -----------------------------------------------------------
# STEP 3: COMPUTE MOVIE–MOVIE COSINE SIMILARITY
# -----------------------------------------------------------

# Cosine similarity tells us how similar two movies are,
# based on how users rated them (ratings pattern).
# A higher score = more similar.
similarity_df = pd.DataFrame(
    cosine_similarity(item_user_matrix),
    index=item_user_matrix.index,     # movieIds as rows
    columns=item_user_matrix.index    # movieIds as columns
)

# -----------------------------------------------------------
# STEP 4: RECOMMENDATION FUNCTION
# -----------------------------------------------------------

def recommend_movies(user_likes, selected_genre, top_n=10):
    """
    This function takes:
      - user_likes: list of movieIds the user likes
      - selected_genre: genre chosen by the user
      - top_n: how many movies to recommend

    It returns the top-N recommended movies using:
      1. Cosine similarity between movies
      2. User-selected genre filtering
    """
    all_scores = {}

    # For each movie the user likes → find similar movies
    for movie_id in user_likes:
        if movie_id not in similarity_df.index:
            continue

        # Get similarity scores for all movies vs the liked movie
        sim_scores = similarity_df[movie_id]

        # Accumulate similarity scores across all liked movies
        for sim_movie_id, score in sim_scores.items():
            # Avoid recommending the same movies user already liked
            if sim_movie_id in user_likes:
                continue

            all_scores[sim_movie_id] = all_scores.get(sim_movie_id, 0) + score

    # Sort movies based on total similarity score
    sorted_movies = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    # Pick top 50 candidates before applying genre filter
    top_ids = [m for m, _ in sorted_movies[:50]]

    # Convert IDs to actual movie titles
    candidates = movies[movies['movieId'].isin(top_ids)]

    # If a genre is selected → filter only those movies
    if selected_genre != "All":
        candidates = candidates[candidates['genres'].apply(lambda x: selected_genre in x)]

    # Return final top-N recommendations
    return candidates.head(top_n)

# -----------------------------------------------------------
# STEP 5: STREAMLIT USER INTERFACE
# -----------------------------------------------------------

st.title("🎬 Movie Recommendation System")
st.write("Select movies you like and a genre to get personalized recommendations.")

# Dictionary of movieId → title (for dropdown)
movie_dict = movies.set_index("movieId")["title"].to_dict()

# Multi-select dropdown for choosing liked movies
selected_likes = st.multiselect(
    "🎞️ Pick movies you like:",
    options=list(movie_dict.keys()),
    format_func=lambda x: movie_dict[x]
)

# Extract all genres from dataset (unique + sorted)
all_genres = sorted(list({
    g
    for genre_list in movies['genres']
    for g in genre_list
}))

# Genre selection dropdown
selected_genre = st.selectbox("🎭 Pick a genre:", ["All"] + all_genres)

# When user clicks "Recommend", run the recommender
if st.button("Recommend"):
    if len(selected_likes) == 0:
        st.warning("⚠️ Select at least one movie.")
    else:
        st.subheader("⭐ Recommended for You")

        # Call our recommendation engine
        recs = recommend_movies(selected_likes, selected_genre)

        # Display results in a table
        st.table(recs[['title', 'genres']])
