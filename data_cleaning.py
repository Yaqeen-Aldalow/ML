import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل البيانات
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")
users_df = pd.read_csv("users.csv")

# استكشاف البيانات
print("Movies DataFrame:")
print(movies_df.head())
print("\nRatings DataFrame:")
print(ratings_df.head())
print("\nUsers DataFrame:")
print(users_df.head())

# التحقق من القيم المفقودة
print("\nMissing Values in Movies DataFrame:")
print(movies_df.isnull().sum())
print("\nMissing Values in Ratings DataFrame:")
print(ratings_df.isnull().sum())
print("\nMissing Values in Users DataFrame:")
print(users_df.isnull().sum())

# استبدال القيم المفقودة أو التخلص منها
movies_df.dropna(subset=['title'], inplace=True)
ratings_df.dropna(inplace=True)
users_df.dropna(inplace=True)

# التحقق من أنواع البيانات
print("\nData Types in Movies DataFrame:")
print(movies_df.dtypes)
print("\nData Types in Ratings DataFrame:")
print(ratings_df.dtypes)
print("\nData Types in Users DataFrame:")
print(users_df.dtypes)

# تغيير النوع إذا لزم الأمر
ratings_df['rating'] = ratings_df['rating'].astype(float)
users_df['age'] = users_df['age'].astype(int)

# التحقق من وجود العمود 'movie_id'
print("\nCheck for 'movie_id' column:")
print("'movie_id' in movies_df columns:", 'movie_id' in movies_df.columns)
print("'movie_id' in ratings_df columns:", 'movie_id' in ratings_df.columns)

# استكشاف التوزيع وقيم التقييمات
sns.histplot(ratings_df['rating'], bins=5)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

print("\nRatings Statistics:")
print(ratings_df['rating'].describe())

# دمج البيانات إذا لزم الأمر
merged_df = pd.merge(ratings_df, users_df, on='user_id')
if 'movie_id' in movies_df.columns:
    merged_df = pd.merge(merged_df, movies_df, left_on='movie_id', right_on='id')

print("\nMerged DataFrame:")
print(merged_df.head())

# حفظ البيانات النظيفة
movies_df.to_csv("clean_movies.csv", index=False)
ratings_df.to_csv("clean_ratings.csv", index=False)
users_df.to_csv("clean_users.csv", index=False)
merged_df.to_csv("clean_merged.csv", index=False)

print("Data cleaning and saving completed.")
