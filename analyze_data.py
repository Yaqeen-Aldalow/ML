import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# تحميل البيانات النظيفة
movies_df = pd.read_csv("clean_movies.csv")
ratings_df = pd.read_csv("clean_ratings.csv")
users_df = pd.read_csv("clean_users.csv")

# تحضير البيانات لاستخدامها مع surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات إلى مجموعة تدريب واختبار (80% تدريب و 20% اختبار)
trainset, testset = train_test_split(data, test_size=0.2)

# تقييم نموذج Collaborative Filtering (KNN)
knn_model = KNNBasic()
knn_model.fit(trainset)
knn_predictions = knn_model.test(testset)
knn_rmse = accuracy.rmse(knn_predictions)
knn_mae = accuracy.mae(knn_predictions)

# تقييم نموذج Matrix Factorization (SVD)
svd_model = SVD()
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions)
svd_mae = accuracy.mae(svd_predictions)

# تقييم نموذج Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genre_ids'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df.index[movies_df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

def content_based_predictions(testset):
    predictions = []
    for uid, iid, true_ratings in testset:
        try:
            movie_idx = movies_df.index[movies_df['id'] == iid].tolist()[0]
            similar_movies = get_recommendations(movies_df['title'][movie_idx])
            predicted_rating = 0
            count = 0
            for idx in similar_movies:
                predicted_rating += ratings_df[(ratings_df['movie_id'] == movies_df.iloc[idx]['id'])]['rating'].mean()
                count += 1
            if count > 0:
                predicted_rating /= count
        except:
            predicted_rating = 3  # افتراض تصنيف افتراضي في حال عدم العثور على توصية
        predictions.append((uid, iid, true_ratings, predicted_rating, None))  # إضافة None لتكملة القيم المطلوبة
    return predictions

content_based_predictions_result = content_based_predictions(testset)
content_based_rmse = accuracy.rmse(content_based_predictions_result)
content_based_mae = accuracy.mae(content_based_predictions_result)

# طباعة النتائج
print(f"RMSE for Collaborative Filtering (KNN): {knn_rmse}")
print(f"MAE for Collaborative Filtering (KNN): {knn_mae}")
print(f"RMSE for Matrix Factorization (SVD): {svd_rmse}")
print(f"MAE for Matrix Factorization (SVD): {svd_mae}")
print(f"RMSE for Content-Based Filtering: {content_based_rmse}")
print(f"MAE for Content-Based Filtering: {content_based_mae}")
