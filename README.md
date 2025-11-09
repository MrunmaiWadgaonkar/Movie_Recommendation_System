# Movie_Recommendation_System

## Team Member: Mrunmai Wadgaonkar
## Reg No: 2023bit049
## Roll No: I46


# 📌 Overview

This project is a **Movie Recommendation System** that suggests similar movies based on metadata such as genres, keywords, cast, and movie overview. 
It uses **content-based filtering** with TF-IDF vectorization and **cosine similarity** to find movies most similar to a user-selected film.


---------------------------------------------------------------------------------------------------------------------------------------------------

# 🗂️ Folder Structure

Movie_Recommendation_System/
│
├── app.py
├── movie-recommender-system.ipynb
├── movie_list.pkl
├── requirements.txt
│
├── data/
│   └── tmdb_5000_movies.csv
│
└── README.md

Note: Large dataset files (`similarity.pkl`, `tmdb_5000_credits.csv`) are excluded due to GitHub size limits.

---------------------------------------------------------------------------------------------------------------------------------------------------

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Cosine Similarity (Content-based Recommendation)

---------------------------------------------------------------------------------------------------------------------------------------------------

# 📊 Dataset
TMDB 5000 Movie Metadata  
Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Included in repository:  
✅ `tmdb_5000_movies.csv` placed in `/data/`

User must manually download:  
❌ `tmdb_5000_credits.csv`

---------------------------------------------------------------------------------------------------------------------------------------------------

# 🚀 Features

✔ Improve recommendation accuracy using advanced NLP (BERT, Sentence Transformers)
✔ Add user-based collaborative filtering module
✔ Deploy with a proper backend + CDN for posters
✔ Add user login + saved watchlist
✔ Allow rating system to refine recommendations

---------------------------------------------------------------------------------------------------------------------------------------------------

# My Contribution
I implemented:
-> Complete API integration and request handling
-> Text preprocessing and tag generation logic
-> TF-IDF model creation and similarity scoring
-> Streamlit frontend and caching optimization
-> Secure deployment setup (no hardcoded API keys)

---------------------------------------------------------------------------------------------------------------------------------------------------

Link: https://movierecommendationsystem-3shahpcn4tegqfq72nss6p.streamlit.app/#recommendations-for-our-fault

