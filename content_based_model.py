import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

movies_df = pd.read_csv("clean_movies.csv")

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

all_recommendations = []

for title in movies_df['title']:
    try:
        recommendations = get_recommendations(title)
        all_recommendations.extend(recommendations)
    except Exception as e:
        print(f"Error with title {title}: {e}")

recommendations_df = pd.DataFrame(all_recommendations, columns=['title'])
recommendations_count = recommendations_df['title'].value_counts()

top_recommendations = recommendations_count.head(20)
plt.figure(figsize=(12, 6))
top_recommendations.plot(kind='bar', edgecolor='k', width=0.8)
plt.title('Top 20 Recommended Movies')
plt.xlabel('Movies')
plt.ylabel('Frequency of Recommendation')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# تحميل بيانات التقييمات
ratings_df = pd.read_csv("clean_ratings.csv")

recommended_df = pd.merge(
    pd.DataFrame(all_recommendations, columns=["title"]),
    movies_df[['title', 'id']],
    on='title',
    how='left'
).rename(columns={'id': 'movie_id'})

print(f"Number of movies in it: {len(recommended_df)}")

# دمج التوصيات مع التقييمات بناءً على movie_id
merged_df = pd.merge(ratings_df, recommended_df, on='movie_id', how='inner')
print(f"Number of recommended and rated movies: {merged_df.shape[0]}")

# حساب True Positives: الأفلام الموصى بها وتقييمها > 3
true_positive = merged_df[merged_df['rating'] > 3].shape[0]

# حساب False Positives: الأفلام الموصى بها وتقييمها ≤ 3
false_positive = merged_df[merged_df['rating'] <= 3].shape[0]

# حساب False Negatives: الأفلام التي لم يُوصى بها لكنها تقييمها > 3
recommended_movie_ids = recommended_df['movie_id'].unique()
false_negative = ratings_df[
    (ratings_df['rating'] > 3) & (~ratings_df['movie_id'].isin(recommended_movie_ids))
].shape[0]

print(f"True Positive: {true_positive}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}")

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
