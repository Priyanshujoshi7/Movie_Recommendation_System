🎬 Movie Recommendation System (Content-Based)

This project builds a content-based movie recommendation system that suggests similar movies using cosine similarity computed from user–item interactions. By analysing how users rate movies, the system identifies patterns and recommends movies that are most similar to a selected movie.

🔍 Key Features

Content-Based Filtering using cosine similarity

User–Item Matrix Construction

Movie–Movie Similarity Matrix Generation

Top-N Recommendations for any given movie

Clean, reusable functions for extracting similar movies

Fully implemented using Python, Pandas, NumPy, and Scikit-Learn

🧠 How It Works

Build a user-item matrix from ratings data

Transpose it into an item-user matrix (movies × users)

Compute cosine similarity between all movies

Store results in a movie similarity DataFrame

Use get_similar_movies() to return recommended titles


📚 Dataset Used

Typically based on MovieLens (ratings + movies metadata), but any structured movie–ratings dataset can be plugged in.

🚀 Tech Stack

Python

Pandas

NumPy

Scikit-Learn

Jupyter Notebook / Google Colab

