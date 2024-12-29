import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
ratings_df = pd.read_csv("clean_ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

svd_model = SVD()
svd_model.fit(trainset)

svd_predictions = svd_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions)
svd_mae = accuracy.mae(svd_predictions)

print(f"RMSE for Matrix Factorization (SVD): {svd_rmse}")
print(f"MAE for Matrix Factorization (SVD): {svd_mae}")

def get_svd_recommendations(user_id, top_n=10):
    all_movie_ids = ratings_df['movie_id'].unique()
    rated_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id']
    un_rated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    predictions = [svd_model.predict(user_id, movie_id) for movie_id in un_rated_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_predictions = predictions[:top_n]
    recommended_movie_ids = [pred.iid for pred in top_predictions]
    return recommended_movie_ids

print(get_svd_recommendations(1)) 