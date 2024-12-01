import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_users(num_users=1000):
    """
    إنشاء بيانات المستخدمين البسيطة
    """
    users = []
    for user_id in range(1, num_users + 1):
        user = {
            'user_id': user_id,
            'age': random.randint(18, 70),
            'join_date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        }
        users.append(user)
    
    return pd.DataFrame(users)

def generate_ratings(users_df, movies_df, min_ratings_per_user=5, max_ratings_per_user=20):
    """
    إنشاء بيانات التقييمات
    """
    ratings = []
    
    for user_id in users_df['user_id']:
        # عدد عشوائي من التقييمات لكل مستخدم
        num_ratings = random.randint(min_ratings_per_user, max_ratings_per_user)
        
        # اختيار أفلام عشوائية للتقييم
        movies_to_rate = movies_df.sample(n=min(num_ratings, len(movies_df)))
        
        for _, movie in movies_to_rate.iterrows():
            rating = {
                'user_id': user_id,
                'movie_id': movie['id'],
                'rating': random.randint(1, 5),  # تقييم من 1 إلى 5
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
            }
            ratings.append(rating)
    
    return pd.DataFrame(ratings)

def create_dataset(movies_file='movies.csv', num_users=100):
    """
    إنشاء مجموعة البيانات الكاملة
    """
    try:
        # قراءة بيانات الأفلام
        movies_df = pd.read_csv(movies_file)
        
        # إنشاء بيانات المستخدمين
        users_df = generate_users(num_users)
        
        # إنشاء بيانات التقييمات
        ratings_df = generate_ratings(users_df, movies_df)
        
        # حفظ البيانات
        users_df.to_csv('users.csv', index=False)
        ratings_df.to_csv('ratings.csv', index=False)
        
        print(f"تم إنشاء {len(users_df)} مستخدم")
        print(f"تم إنشاء {len(ratings_df)} تقييم")
        
        return True
        
    except Exception as e:
        print(f"حدث خطأ: {str(e)}")
        return False

# تشغيل الكود
if __name__ == "__main__":
    create_dataset()