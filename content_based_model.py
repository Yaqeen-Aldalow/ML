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

print(get_recommendations('Deadpool & Wolverine'))
recommendations = get_recommendations('Deadpool & Wolverine')
plt.hist(recommendations, bins=10, edgecolor='k')
plt.show() 