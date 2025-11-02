# app.py
import streamlit as st
import requests
import os
from typing import List, Dict, Optional

# -------------------------
# Configuration / secrets
# -------------------------
def get_api_key() -> Optional[str]:
    # First try Streamlit secrets (when deployed)
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        # st.secrets may raise if not available locally
        pass
    # Fallback to environment variable for local dev
    return os.getenv("TMDB_API_KEY")

API_KEY = get_api_key()
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if not API_KEY:
    st.title("Movie Recommender")
    st.error(
        "TMDB API key not found. Add `TMDB_API_KEY` to Streamlit Secrets (recommended) "
        "or export it as an environment variable locally."
    )
    st.info("Streamlit Cloud: Manage app → Secrets. Locally: export TMDB_API_KEY='your_key'")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def tmdb_get(path: str, params: dict = None, retries: int = 2) -> dict:
    """GET request to TMDB with simple retry logic."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < retries:
                continue
            # return empty dict so caller can handle gracefully
            return {}

def search_movies(query: str, page: int = 1) -> List[Dict]:
    data = tmdb_get("/search/movie", {"query": query, "page": page})
    return data.get("results", []) if isinstance(data, dict) else []

def fetch_similar_movies(movie_id: int, topn: int = 5) -> List[Dict]:
    data = tmdb_get(f"/movie/{movie_id}/similar", {"page": 1})
    results = data.get("results", []) if isinstance(data, dict) else []
    out = []
    for r in results[:topn]:
        out.append({
            "title": r.get("title") or r.get("name"),
            "poster": IMAGE_BASE + r["poster_path"] if r.get("poster_path") else None,
            "id": r.get("id"),
            "release_date": r.get("release_date") or r.get("first_air_date", "")
        })
    return out

def fetch_movie_by_id(movie_id: int) -> Optional[Dict]:
    d = tmdb_get(f"/movie/{movie_id}")
    if not d:
        return None
    return {
        "title": d.get("title"),
        "id": d.get("id"),
        "poster": IMAGE_BASE + d["poster_path"] if d.get("poster_path") else None,
        "overview": d.get("overview", ""),
        "release_date": d.get("release_date", "")
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender — Search & Similar")

st.write("Type a movie name, press **Search**, pick the exact match (if multiple), and see 5 similar movies.")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Search movie title (e.g. Inception)", value="")
with col2:
    search_btn = st.button("Search")

# store last chosen movie in session_state so we can display consistently
if "selected_movie_id" not in st.session_state:
    st.session_state["selected_movie_id"] = None
if "selected_movie_title" not in st.session_state:
    st.session_state["selected_movie_title"] = None

if search_btn and query:
    with st.spinner("Searching TMDB..."):
        results = search_movies(query)
    if not results:
        st.warning("No results found on TMDB. Try a different query.")
    else:
        # prepare display options
        options = []
        id_map = {}
        for r in results[:10]:
            title = r.get("title") or r.get("name")
            year = (r.get("release_date") or r.get("first_air_date") or "")[:4]
            label = f"{title} ({year})" if year else title
            options.append(label)
            id_map[label] = r.get("id")
        choice = st.selectbox("Select exact movie", ["-- pick one --"] + options)
        if choice != "-- pick one --":
            chosen_id = id_map.get(choice)
            st.session_state["selected_movie_id"] = chosen_id
            st.session_state["selected_movie_title"] = choice

# If a movie is selected (from selectbox or previous session state), fetch similar
movie_id = st.session_state.get("selected_movie_id")
movie_title = st.session_state.get("selected_movie_title")

if movie_id:
    # Show chosen movie header
    with st.spinner("Fetching movie details..."):
        base = fetch_movie_by_id(movie_id)
    if base:
        st.subheader(f"Recommendations for: {base['title']} {f'({base.get(\"release_date\",\"\")[:4]})' if base.get('release_date') else ''}")
        # fetch similar movies
        with st.spinner("Fetching similar movies from TMDB..."):
            recs = fetch_similar_movies(movie_id, topn=5)
        if not recs:
            st.info("TMDB returned no similar movies. Try another title or search again.")
        else:
            # Grid view: 5 columns
            cols = st.columns(5)
            for i, rec in enumerate(recs):
                with cols[i]:
                    st.text(rec["title"])
                    if rec["poster"]:
                        st.image(rec["poster"], use_column_width=True)
                    else:
                        st.write("Poster unavailable")
                    if rec.get("release_date"):
                        st.caption(rec["release_date"][:10])
    else:
        st.error("Could not fetch movie details. Try again.")

# Footer / tips
st.markdown("---")
st.write("Tips: If you don't see good matches in the popular pool, try a more exact search (include year).")
