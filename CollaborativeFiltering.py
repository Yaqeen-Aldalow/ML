from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic, SVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# تحميل البيانات من ملف CSV
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات إلى تدريب واختبار (80% تدريب و 20% اختبار)
trainset, testset = train_test_split(data, test_size=0.2)

# نموذج Collaborative Filtering باستخدام KNNBasic
knn_basic_model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
knn_basic_model.fit(trainset)
knn_basic_predictions = knn_basic_model.test(testset)
knn_basic_rmse = accuracy.rmse(knn_basic_predictions)
knn_basic_mae = accuracy.mae(knn_basic_predictions)

# نموذج Matrix Factorization باستخدام SVD
svd_model = SVD()
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions)
svd_mae = accuracy.mae(svd_predictions)

# نموذج Content-Based Filtering (باستخدام Cosine Similarity)
# نحتاج إلى بيانات إضافية مثل العناوين أو الوصف
# هنا نفترض أن `movies_df` يحتوي على العمود 'title' الذي يحتوي على العناوين

# تحويل العناوين إلى تمثيل TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title'])

# حساب التشابه باستخدام Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# دالة للتوصية بناءً على Content-Based Filtering
def content_based_recommendation(movie_id, cosine_sim=cosine_sim):
    idx = movies_df.index[movies_df['id'] == movie_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # أفضل 10 أفلام مشابهة
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

ratings_df['content_based_recommendations'] = ratings_df['movie_id'].apply(content_based_recommendation)

# طباعة النتائج:
print(f"RMSE for Collaborative Filtering (KNNBasic): {knn_basic_rmse}")
print(f"MAE for Collaborative Filtering (KNNBasic): {knn_basic_mae}")
print(f"RMSE for Matrix Factorization (SVD): {svd_rmse}")
print(f"MAE for Matrix Factorization (SVD): {svd_mae}")
print("Content-Based Recommendations for the first few ratings:")
print(ratings_df.head()[['movie_id', 'content_based_recommendations']])
