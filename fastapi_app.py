import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# قراءة بيانات التقييمات من ملف CSV (مثال)
ratings_df = pd.read_csv("path_to_your_ratings_file.csv")

# عرض أول خمس صفوف للتأكد من البيانات
print(ratings_df.head())

# إعداد البيانات باستخدام مكتبة Surprise
reader = Reader(rating_scale=(1, 5))  # تحديد مقياس التقييم (من 1 إلى 5)
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات إلى تدريب واختبار
trainset, testset = train_test_split(data, test_size=0.2)

# استخدام خوارزمية SVD (Matrix Factorization)
svd = SVD()

# تدريب النموذج
svd.fit(trainset)

# حساب دقة النموذج باستخدام RMSE
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# توليد التوصيات للمستخدم (مثال user_id = 1)
def generate_recommendations(user_id, n_recommendations=10):
    # توليد جميع التوقعات للأفلام التي لم يقيمها المستخدم
    all_movie_ids = ratings_df['movie_id'].unique()
    movie_ids_rated_by_user = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in movie_ids_rated_by_user]
    
    # توقع التقييمات لكل الأفلام
    predictions = [svd.predict(user_id, movie_id) for movie_id in movies_to_predict]
    
    # ترتيب التوقعات حسب التقييم المتوقع
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # أخذ أفضل N توصيات
    top_n_recommendations = predictions[:n_recommendations]
    
    # إرجاع أسماء الأفلام
    recommended_movie_ids = [pred.iid for pred in top_n_recommendations]
    recommended_movies = ratings_df[ratings_df['movie_id'].isin(recommended_movie_ids)]['movie_title'].unique()
    
    return {"movies": recommended_movies}

# اختبار التوصيات للمستخدم 1
user_id = 1
recommended_movies = generate_recommendations(user_id, n_recommendations=10)
print("Recommended Movies:", recommended_movies)
