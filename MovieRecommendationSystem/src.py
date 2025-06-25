import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Set the page config for Streamlit
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Load the data with the correct paths
@st.cache_data  # Cache the data for better performance
def load_data():
    # Read the data from the ml-latest-small directory
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, 
                                index=user_item_matrix.index, 
                                columns=user_item_matrix.index)

def get_recommendations(user_id, n=5):
    try:
        # Get similar users (excluding the user themselves)
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11].index
        
        # Get movies rated by similar users
        similar_users_ratings = user_item_matrix.loc[similar_users]
        
        # Calculate average ratings
        avg_ratings = similar_users_ratings.mean(axis=0)
        
        # Get movies user hasn't rated
        user_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
        avg_ratings = avg_ratings.drop(user_rated, errors='ignore')
        
        # Get top n recommendations with movie titles
        recommendations = avg_ratings.sort_values(ascending=False).head(n)
        recommended_movies = []
        for movie_id in recommendations.index:
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            recommended_movies.append((movie_title, recommendations[movie_id]))
        
        return recommended_movies
    except KeyError:
        return None

# Streamlit UI
st.title('🎬 Movie Recommendation System')
st.write("This system recommends movies based on collaborative filtering using the MovieLens dataset.")

# User selection
col1, col2 = st.columns([1, 3])
with col1:
    user_id = st.selectbox('Select User ID', sorted(ratings['userId'].unique()))
    n_recommendations = st.slider('Number of recommendations', 1, 10, 5)

with col2:
    st.write("### How it works:")
    st.write("1. The system finds users with similar rating patterns to the selected user")
    st.write("2. It then recommends movies that these similar users rated highly")
    st.write("3. Only movies the selected user hasn't rated are considered")

# Display recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_id, n_recommendations)
    
    if recommendations:
        st.success(f"Top {n_recommendations} recommendations for User {user_id}:")
        
        # Display recommendations in a nice format
        for i, (movie, predicted_rating) in enumerate(recommendations, 1):
            st.markdown(f"""
            **{i}. {movie}**  
            Predicted rating: {predicted_rating:.2f}/5  
            """)
            
        # Show some stats
        user_ratings = ratings[ratings['userId'] == user_id]
        st.write(f"\nUser {user_id} has rated {len(user_ratings)} movies.")
    else:
        st.error("Could not generate recommendations for this user. Please try another user ID.")

# Optional: Show raw data
if st.checkbox('Show raw data preview'):
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Movies Data")
        st.write(movies.head())
    with col2:
        st.write("### Ratings Data")
        st.write(ratings.head())