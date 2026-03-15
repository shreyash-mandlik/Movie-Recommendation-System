import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# Page config
# ================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .movie-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    .movie-title {
        color: white;
        font-size: 14px;
        font-weight: bold;
    }
    .movie-rating {
        color: #ffd700;
        font-size: 12px;
    }
    .similarity {
        color: #00ff88;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    # Load movies data
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags'])
    similarity = cosine_similarity(vectors)
    
    return similarity, movies
# This line is critical!
similarity, movies = load_model()

# ================================
# TMDB API — fetch poster
# ================================
API_KEY = "56f9561518f7a35420931a4f989f6e3c"  # Replace with your API key

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# ================================
# Recommendation function
# ================================
def recommend(movie_title, n=10):
    matches = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
    if matches.empty:
        return None
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            'id': movies.loc[i, 'id'],
            'title': movies.loc[i, 'title'],
            'similarity': round(score*100, 1),
            'rating': movies.loc[i, 'vote_average'],
        })
    return recommendations

# ================================
# App UI
# ================================
st.title("🎬 Movie Recommendation System")
st.write("Discover movies you'll love based on your favorites!")

# Search box
col1, col2 = st.columns([3, 1])
with col1:
    movie_input = st.text_input("🔍 Enter a movie name:", 
                                 placeholder="e.g. Avatar, The Dark Knight, Avengers...")
with col2:
    n_recommendations = st.selectbox("Number of recommendations:", [5, 10, 15])

# Recommend button
if st.button("🎯 Get Recommendations", type="primary"):
    if movie_input:
        with st.spinner('Finding similar movies...'):
            results = recommend(movie_input, n_recommendations)
        
        if results is None:
            st.error(f"Movie '{movie_input}' not found! Try another name.")
        else:
            st.success(f"Top {n_recommendations} movies similar to '{movie_input}'")
            st.divider()
            
            # Display in grid — 5 per row
            cols = st.columns(5)
            for idx, movie in enumerate(results):
                with cols[idx % 5]:
                    poster = fetch_poster(movie['id'])
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ {movie['rating']} | 🎯 {movie['similarity']}% match")
    else:
        st.warning("Please enter a movie name!")

# ================================
# Popular movies section
# ================================
st.divider()
st.subheader("🔥 Most Popular Movies in Dataset")

top_movies = movies.nlargest(10, 'popularity')[['title', 'vote_average', 'popularity']]
top_movies.columns = ['Title', 'Rating', 'Popularity Score']
top_movies['Popularity Score'] = top_movies['Popularity Score'].round(1)
st.dataframe(top_movies, hide_index=True, use_container_width=True)

