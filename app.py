# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ----------------------
# Configuration / Secrets
# ----------------------
# Streamlit Cloud: use st.secrets["TMDB_API_KEY"]
# Local dev: use .env or export TMDB_API_KEY in shell
API_KEY = None
if "TMDB_API_KEY" in st.secrets:
    API_KEY = st.secrets["TMDB_API_KEY"]
else:
    API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    st.error("TMDB API key not found. Set TMDB_API_KEY in Streamlit secrets or as an environment variable.")
    st.stop()

BASE_URL = "https://api.themoviedb.org/3"

# ----------------------
# Utilities
# ----------------------
def tmdb_get(path, params=None, retries=2, backoff=0.3):
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"
    for attempt in range(retries+1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff * (attempt+1))
                continue
            raise

def fetch_poster_path(movie):
    poster_path = movie.get("poster_path")
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# minimal text cleaning
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------
# Data preparation & caching
# ----------------------
@st.cache_data(show_spinner=False)
def fetch_popular_movies(max_pages=3):
    """
    Fetch popular movies pages using TMDB. Default fetches 3 pages (~60 movies).
    Increase max_pages if you want a larger pool (but beware of API limits/time).
    """
    movies = []
    for page in range(1, max_pages + 1):
        data = tmdb_get("/movie/popular", {"page": page})
        movies.extend(data.get("results", []))
        time.sleep(0.1)
    return movies

@st.cache_data(show_spinner=False)
def enrich_movie_details(movie_ids):
    """
    Given a list of TMDB movie IDs, fetch details with credits and keywords (append_to_response).
    Returns list of dicts with combined metadata and poster path.
    """
    enriched = []
    for mid in movie_ids:
        # Use append_to_response to reduce number of calls
        path = f"/movie/{mid}"
        params = {"append_to_response": "credits,keywords"}
        try:
            d = tmdb_get(path, params)
        except Exception:
            # If a single fetch fails, skip gracefully
            continue

        # build tags: genres, overview, keywords, top 5 cast, director
        genres = [g.get("name","") for g in d.get("genres", [])]
        overview = d.get("overview", "") or ""
        keywords = [k.get("name","") for k in d.get("keywords", {}).get("keywords", [])]
        # credits
        cast_list = d.get("credits", {}).get("cast", [])[:5]
        cast = [c.get("name","") for c in cast_list]
        crew = d.get("credits", {}).get("crew", [])
        director = [member.get("name","") for member in crew if member.get("job","") == "Director"]
        tags = " ".join(genres + keywords + cast + director + [overview])
        enriched.append({
            "movie_id": d.get("id"),
            "title": d.get("title"),
            "tags": clean_text(tags),
            "overview": overview,
            "poster": fetch_poster_path(d)
        })
        time.sleep(0.08)  # respectful pause
    return enriched

@st.cache_resource(show_spinner=False)
def build_model(enriched_movies):
    """
    Build TF-IDF matrix and cosine similarity on 'tags'.
    Returns (df, tfidf_vectorizer, similarity_matrix)
    """
    df = pd.DataFrame(enriched_movies).drop_duplicates(subset=["movie_id"])
    if df.empty:
        return df, None, None
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"].fillna(""))
    sim = cosine_similarity(vectors)
    return df.reset_index(drop=True), tfidf, sim

# ----------------------
# Recommendation functions
# ----------------------
def recommend_from_pool(df, sim_matrix, title, topn=5):
    if df is None or sim_matrix is None or title is None:
        return []
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        # try partial match
        matches = df[df["title"].str.lower().str.contains(title.lower())]
        if matches.empty:
            return []
    idx = matches.index[0]
    distances = list(enumerate(sim_matrix[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in distances[1: topn+1]:
        recommendations.append({
            "title": df.loc[i, "title"],
            "movie_id": df.loc[i, "movie_id"],
            "poster": df.loc[i, "poster"],
            "score": float(score)
        })
    return recommendations

def tmdb_search_movies(query, page=1):
    data = tmdb_get("/search/movie", {"query": query, "page": page})
    return data.get("results", [])

def tmdb_fetch_similar(movie_id, topn=5):
    data = tmdb_get(f"/movie/{movie_id}/similar", {"page":1})
    results = data.get("results", [])[:topn]
    out = []
    for r in results:
        out.append({
            "title": r.get("title"),
            "movie_id": r.get("id"),
            "poster": fetch_poster_path(r)
        })
    return out

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Content-Based Movie Recommender")

# Sidebar controls: choose pool size
with st.sidebar:
    st.header("Settings")
    page_count = st.number_input("Popular-pages to fetch (each page ~20 movies)", min_value=1, max_value=10, value=3, step=1)
    st.caption("Increasing pages gives a larger candidate pool but takes more time and API calls.")
    refresh = st.button("Refresh cached data (use if you changed pages)")

if refresh:
    # clear caches and rerun
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# Fetch popular movies for dropdown pool
with st.spinner("Fetching popular movies..."):
    popular = fetch_popular_movies(max_pages=int(page_count))
    popular_ids = [m["id"] for m in popular]

with st.spinner("Enriching movie metadata (this may take a few seconds)..."):
    enriched = enrich_movie_details(popular_ids)

df, tfidf_model, sim_matrix = build_model(enriched)

# Prepare UI inputs: dropdown list from df and search bar
popular_titles = list(df["title"]) if df is not None else []
col1, col2 = st.columns([3, 2])
with col1:
    selected_movie = st.selectbox("Select a movie from popular list", options=[""] + popular_titles)
with col2:
    query = st.text_input("Or search any movie title (then press Search)")

search_btn = st.button("Search")
if search_btn and query:
    results = tmdb_search_movies(query)
    if not results:
        st.warning("No results from TMDB for your query.")
    else:
        # show top 5 results and let user pick
        options = {f'{r["title"]} ({r.get("release_date","")[:4] if r.get("release_date") else ""})': r for r in results[:7]}
        choice = st.selectbox("Search results", options.keys())
        chosen = options[choice]
        selected_movie = chosen["title"]
        # Also try to fetch similar directly from TMDB if not in pool
        sim_from_tmdb = tmdb_fetch_similar(chosen["id"], topn=5)
        if sim_from_tmdb:
            st.subheader(f"Recommendations for {chosen['title']} (TMDB similar):")
            cols = st.columns(5)
            for i, rec in enumerate(sim_from_tmdb):
                with cols[i]:
                    st.text(rec["title"])
                    if rec["poster"]:
                        st.image(rec["poster"], use_column_width=True)
            st.stop()

# If user selected from dropdown or found via search, produce recommendations
if selected_movie:
    # prefer pool-based recommendation (content-based) if title in df
    if not df.empty and any(df["title"].str.lower() == selected_movie.lower()):
        recs = recommend_from_pool(df, sim_matrix, selected_movie, topn=5)
        if recs:
            st.subheader(f"Recommendations for {selected_movie}:")
            cols = st.columns(5)
            for i, r in enumerate(recs):
                with cols[i]:
                    st.text(r["title"])
                    if r["poster"]:
                        st.image(r["poster"], use_column_width=True)
                    else:
                        st.write("Poster unavailable")
        else:
            st.warning("No recommendations found in local pool.")
    else:
        # fallback: use TMDB similar endpoint (if movie not in pool)
        # first search to get movie id
        sr = tmdb_search_movies(selected_movie)
        if sr:
            mid = sr[0]["id"]
            recs = tmdb_fetch_similar(mid, topn=5)
            if recs:
                st.subheader(f"Recommendations for {selected_movie} (via TMDB similar):")
                cols = st.columns(5)
                for i, r in enumerate(recs):
                    with cols[i]:
                        st.text(r["title"])
                        if r["poster"]:
                            st.image(r["poster"], use_column_width=True)
                        else:
                            st.write("Poster unavailable")
            else:
                st.warning("TMDB returned no similar movies.")
        else:
            st.warning("Could not find the movie on TMDB. Try a different query.")
