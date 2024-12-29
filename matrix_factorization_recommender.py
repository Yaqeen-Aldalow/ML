import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

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

def get_all_predictions():
    all_predictions = []
    
    for user_id in ratings_df['user_id'].unique():
        for movie_id in ratings_df['movie_id'].unique():
            prediction = svd_model.predict(user_id, movie_id)
            all_predictions.append(prediction.est)  # نضيف التنبؤ بالتقييم
            
    return all_predictions

all_predictions = get_all_predictions()

plt.hist(all_predictions, bins=20, edgecolor='k')
plt.title("Distribution of All Movie Recommendations")
plt.xlabel("Predicted Rating")
plt.ylabel("Frequency")
plt.show()

def precision_recall_at_k(predictions, ratings_df, k=10):
    relevant_items = []
    recommended_items = []

    for prediction in predictions:
        user_id = prediction.uid
        movie_id = prediction.iid
        actual_rating = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)]['rating'].values
        if len(actual_rating) > 0:
            actual_rating = actual_rating[0]
            if actual_rating >= 4:
                relevant_items.append(movie_id)
                recommended_items.append(movie_id)

    precision = len(set(relevant_items).intersection(recommended_items)) / len(recommended_items) if recommended_items else 0
    recall = len(set(relevant_items).intersection(recommended_items)) / len(relevant_items) if relevant_items else 0
    
    return precision, recall

predictions_for_precision_recall = [svd_model.predict(pred.uid, pred.iid) for pred in svd_predictions]

precision, recall = precision_recall_at_k(predictions_for_precision_recall, ratings_df)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
