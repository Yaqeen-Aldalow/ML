import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic

# تحميل البيانات النظيفة
ratings_df = pd.read_csv("clean_ratings.csv")

# تحضير البيانات لاستخدامها مع surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# تقسيم البيانات إلى مجموعة تدريب واختبار (80% تدريب و 20% اختبار)
trainset, testset = train_test_split(data, test_size=0.2)

# تطبيق نموذج KNNBasic
knn_model = KNNBasic()
knn_model.fit(trainset)

# تقييم النموذج
knn_predictions = knn_model.test(testset)
knn_rmse = accuracy.rmse(knn_predictions)
knn_mae = accuracy.mae(knn_predictions)

print(f"RMSE for Collaborative Filtering (KNN): {knn_rmse}")
print(f"MAE for Collaborative Filtering (KNN): {knn_mae}")
