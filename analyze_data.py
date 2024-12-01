import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# قراءة البيانات من ملفات CSV
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")
users_df = pd.read_csv("users.csv")

# تحضير البيانات لاستخدامها مع surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات إلى مجموعة تدريب واختبار (80% تدريب و 20% اختبار)
trainset, testset = train_test_split(data, test_size=0.2)

# تدريب نماذج Collaborative Filtering (KNN)
knn_model = KNNBasic()
knn_model.fit(trainset)

# تدريب نموذج Matrix Factorization (SVD)
svd_model = SVD()
svd_model.fit(trainset)

# لتحضير بيانات Content-Based Filtering باستخدام TF-IDF
# تحويل الأنواع (genre) إلى تمثيل عددي باستخدام TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
movies_df['genre_ids'] = movies_df['genre_ids'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genre_ids'])

# حساب التشابه باستخدام Cosine Similarity بين الأفلام
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# دالة للحصول على التنبؤات بناءً على Content-Based Filtering
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df.index[movies_df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices

# حساب التنبؤات لـ Content-Based Filtering (باستخدام تشابه Cosine)
def content_based_predictions(testset):
    predictions = []
    for uid, iid, true_ratings in testset:
        try:
            movie_idx = movies_df.index[movies_df['movie_id'] == iid].tolist()[0]
            similar_movies = get_recommendations(movies_df['title'][movie_idx])
            predicted_rating = 0
            count = 0
            for idx in similar_movies:
                predicted_rating += ratings_df[(ratings_df['movie_id'] == movies_df.iloc[idx]['movie_id'])]['rating'].mean()
                count += 1
            if count > 0:
                predicted_rating /= count
        except:
            predicted_rating = 3  # افتراض تصنيف افتراضي في حال عدم العثور على توصية
        predictions.append((uid, iid, true_ratings, predicted_rating, None))  # إضافة None لتكملة القيم المطلوبة
    return predictions

# إجراء التنبؤات لكل نموذج
knn_predictions = knn_model.test(testset)
svd_predictions = svd_model.test(testset)
content_based_predictions_result = content_based_predictions(testset)

# حساب RMSE و MAE لكل نموذج
knn_rmse = accuracy.rmse(knn_predictions)
svd_rmse = accuracy.rmse(svd_predictions)

knn_mae = accuracy.mae(knn_predictions)
svd_mae = accuracy.mae(svd_predictions)

# حساب RMSE و MAE لنموذج Content-Based Filtering
content_based_rmse = accuracy.rmse(content_based_predictions_result)
content_based_mae = accuracy.mae(content_based_predictions_result)

# طباعة النتائج
print(f"RMSE for Collaborative Filtering (KNN): {knn_rmse}")
print(f"RMSE for Matrix Factorization (SVD): {svd_rmse}")
print(f"RMSE for Content-Based Filtering: {content_based_rmse}")

print(f"MAE for Collaborative Filtering (KNN): {knn_mae}")
print(f"MAE for Matrix Factorization (SVD): {svd_mae}")
print(f"MAE for Content-Based Filtering: {content_based_mae}")
