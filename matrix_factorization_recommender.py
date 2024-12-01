from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# تحميل البيانات
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], Reader(rating_scale=(1, 5)))

# تقسيم البيانات
trainset, testset = train_test_split(data, test_size=0.2)

# تطبيق SVD
model = SVD()

# تدريب النموذج
model.fit(trainset)

# تقييم النموذج
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# وظيفة لتقديم توصيات
def get_svd_recommendations(user_id, top_n=10):
    # التنبؤات للأفلام غير المصنفة للمستخدم
    all_movie_ids = ratings_df['movie_id'].unique()
    rated_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id']
    un_rated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in un_rated_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_predictions = predictions[:top_n]
    recommended_movie_ids = [pred.iid for pred in top_predictions]
    return movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]['title']

# عرض أفضل 10 توصيات لمستخدم معين
print(get_svd_recommendations(1))  # افترض أن user_id هو 1
