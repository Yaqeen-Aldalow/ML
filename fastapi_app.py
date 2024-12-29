from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

movies_df = pd.read_csv("clean_movies.csv")

tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genre_ids'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

class RecommendationRequest(BaseModel):
    title: str

def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies_df.index[movies_df['title'] == title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies_df['title'].iloc[movie_indices].tolist()
    except IndexError:
        return []

@app.post("/recommend/")
async def recommend_movies(req: RecommendationRequest):
    recommendations = get_recommendations(req.title, cosine_sim)
    return {"recommended_movies": recommendations}
