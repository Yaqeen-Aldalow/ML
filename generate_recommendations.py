import pandas as pd

movies_df = pd.read_csv("movies.csv")  
ratings_df = pd.read_csv("ratings.csv")  

def get_recommendations(user_id: int, n_recommendations: int):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    all_movies = movies_df['movie_id'].unique()
    user_watched_movies = user_ratings['movie_id'].unique()
    recommendations = [movie for movie in all_movies if movie not in user_watched_movies]
    return {"recommended_movies": recommendations[:n_recommendations]}
