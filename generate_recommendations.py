import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# تحميل البيانات من ملفات CSV
ratings_df = pd.read_csv("C:/Users/yaqee/Desktop/ML/ML/ratings.csv")
users_df = pd.read_csv("C:/Users/yaqee/Desktop/ML/ML/users.csv")
movies_df = pd.read_csv("C:/Users/yaqee/Desktop/ML/ML/movies.csv")

# إعداد البيانات لتناسب Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات
trainset, testset = train_test_split(data, test_size=0.2)

# تدريب النموذج
model = SVD()
model.fit(trainset)

# دالة للحصول على التوصيات
def get_recommendations(user_id, n_recommendations=10):
    # تحديد التقييمات المتوقعة لجميع الأفلام
    all_movies = movies_df['movie_id'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movies]
    
    # ترتيب التوقعات بناءً على التقييم المتوقع
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # تحديد أفضل التوصيات
    recommended_movie_ids = [pred.iid for pred in predictions[:n_recommendations]]
    
    # الحصول على أسماء الأفلام من DataFrame الأفلام
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]
    
    # إعادة قائمة الأفلام الموصى بها
    return recommended_movies[['movie_id', 'title']].to_dict(orient='records')
