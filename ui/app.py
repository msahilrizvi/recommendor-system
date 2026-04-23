import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
API_URL = "http://127.0.0.1:8000/recommend"
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------
# Load movie metadata
# ----------------------------
@st.cache_data
def load_movies():
    movie_path = BASE_DIR / "data/raw/ml-100k/u.item"
    
    movies = pd.read_csv(
        movie_path,
        sep="|",
        encoding="latin-1",
        header=None
    )
    
    movies = movies[[0, 1]]  # item_id, title
    movies.columns = ["item_id", "title"]
    
    return movies

movies_df = load_movies()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations instantly!")

# Inputs
user_id = st.number_input("Enter User ID", min_value=1, step=1)
n = st.slider("Number of recommendations", 1, 20, 10)

# ----------------------------
# Button
# ----------------------------
if st.button("Get Recommendations"):
    
    params = {"user_id": user_id, "n": n}
    
    try:
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            st.subheader(f"Recommendations for User {user_id}")
            
            # Handle fallback note
            if "note" in data:
                st.warning(data["note"])
            
            recommendations = data.get("recommendations", [])
            
            if not recommendations:
                st.info("No recommendations found.")
            else:
                for item in recommendations:
                    # Map item_id → movie title
                    title_row = movies_df[movies_df["item_id"] == item]
                    
                    if not title_row.empty:
                        title = title_row.iloc[0]["title"]
                        st.markdown(f"🎬 **{title}** (ID: {item})")
                    else:
                        st.markdown(f"🎬 Movie ID: {item}")
        
        else:
            st.error("Error fetching recommendations from API.")
    
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")