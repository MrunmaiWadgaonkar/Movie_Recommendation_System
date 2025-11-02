import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    st.error("TMDB API key not found. Set TMDB_API_KEY in Streamlit secrets or as an environment variable.")
    st.stop()

BASE_URL = "https://api.themoviedb.org/3"

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
        except Exception:
            if attempt < retries:
                time.sleep(backoff * (attempt+1))
                continue
            raise

def fetch_poster_path(movie):
    poster_path = movie.get("poster_path")
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data(show_spinner=False)
def fetch_popular_movies(max_pages=3):
    movies = []
    for page in range(1, max_pages + 1):
        data = tmdb_get("/movie/popular", {"page": page})
        movies.extend(data.get("results", []))
        time.sleep(0.1)
    return movies

@st.cache_data(show_spinner=False)
def enrich_movie_details(movie_ids):
    enriched = []
    for mid in movie_ids:
        path = f"/movie/{mid}"
        params = {"append_to_response": "credits,keywords"}
        try:
            d = tmdb_get(path, params)
        except Exception:
            continue

        genres = [g.get("name","") for g in d.get("genres", [])]
        overview = d.get("overview", "") or ""
        keywords = [k.get("name","") for k in d.get("keywords", {}).get("keywords", [])]
        cast = [c.get("name","") for c in d.get("credits", {}).get("cast", [])[:5]]
        director = [c.get("name","") for c in d.get("credits", {}).get("crew", []) if c.get("job") == "Director"]

        tags = " ".join(genres + keywords + cast + director + [overview])
        enriched.append({
            "movie_id": d.get("id"),
            "title": d.get("title"),
            "tags": clean_text(tags),
            "overview": overview,
            "poster": fetch_poster_path(d)
        })
        time.sleep(0.08)
    return enriched

@st.cache_resource(show_spinner=False)
def build_model(enriched_movies):
    df = pd.DataFrame(enriched_movies).drop_duplicates(subset=["movie_id"])
    if df.empty:
        return df, None, None
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"].fillna(""))
    sim = cosine_similarity(vectors)
    return df.reset_index(drop=True), tfidf, sim

def recommend_from_pool(df, sim_matrix, title, topn=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title.lower())]
        if matches.empty:
            return []
    idx = matches.index[0]
    distances = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)
    recs = []
    for i, score in distances[1: topn+1]:
        recs.append({
            "title": df.loc[i, "title"],
            "movie_id": df.loc[i, "movie_id"],
            "poster": df.loc[i, "poster"]
        })
    return recs

def tmdb_search_movies(query, page=1):
    data = tmdb_get("/search/movie", {"query": query, "page": page})
    return data.get("results", [])

def tmdb_fetch_similar(movie_id, topn=5):
    data = tmdb_get(f"/movie/{movie_id}/similar", {"page":1})
    return [{
        "title": r.get("title"),
        "movie_id": r.get("id"),
        "poster": fetch_poster_path(r)
    } for r in data.get("results", [])[:topn]]

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Content-Based Movie Recommender")

PAGE_COUNT = 3

with st.spinner("Fetching movies..."):
    popular = fetch_popular_movies(max_pages=PAGE_COUNT)
popular_ids = [m["id"] for m in popular]

with st.spinner("Analyzing movies..."):
    enriched = enrich_movie_details(popular_ids)
df, tfidf_model, sim_matrix = build_model(enriched)

popular_titles = list(df["title"]) if df is not None else []

col1, col2 = st.columns([3, 2])
with col1:
    selected_movie = st.selectbox("Pick a popular movie", options=[""] + popular_titles)
with col2:
    query = st.text_input("Or search movie title")

search_btn = st.button("Search")

if search_btn and query:
    results = tmdb_search_movies(query)
    if not results:
        st.warning("No results found.")
    else:
        options = {
            f'{r["title"]} ({(r.get("release_date") or "")[:4]})': r
            for r in results[:7]
        }
        choice = st.selectbox("Select movie", options.keys())
        chosen = options[choice]
        selected_movie = chosen["title"]
        recs = tmdb_fetch_similar(chosen["id"], topn=5)

        st.subheader(f"Recommendations for {chosen['title']} (TMDB Similar)")
        cols = st.columns(5)
        for i, r in enumerate(recs):
            with cols[i]:
                st.text(r["title"])
                if r["poster"]:
                    st.image(r["poster"], use_column_width=True)
        st.stop()

if selected_movie:
    if any(df["title"].str.lower() == selected_movie.lower()):
        recs = recommend_from_pool(df, sim_matrix, selected_movie, topn=5)
        st.subheader(f"Recommendations for {selected_movie}")
        cols = st.columns(5)
        for i, r in enumerate(recs):
            with cols[i]:
                st.text(r["title"])
                if r["poster"]:
                    st.image(r["poster"], use_column_width=True)
    else:
        sr = tmdb_search_movies(selected_movie)
        if sr:
            recs = tmdb_fetch_similar(sr[0]["id"], topn=5)
            st.subheader(f"Recommendations for {selected_movie}")
            cols = st.columns(5)
            for i, r in enumerate(recs):
                with cols[i]:
                    st.text(r["title"])
                    if r["poster"]:
                        st.image(r["poster"], use_column_width=True)
        else:
            st.warning("No results found on TMDB.")
