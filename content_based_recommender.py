from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# تحميل البيانات النظيفة
movies_df = pd.read_csv("clean_movies.csv")

# تحويل الأنواع (genres) إلى تمثيل عددي باستخدام TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genre_ids'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# حساب التشابه باستخدام Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# دالة للحصول على الأفلام الأكثر تشابهًا مع فيلم معين
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies_df.index[movies_df['title'] == title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies_df['title'].iloc[movie_indices]
    except IndexError:
        return []

class RecommendationRequest(BaseModel):
    title: str

@app.post("/recommend/")
async def recommend_movies(req: RecommendationRequest):
    recommendations = get_recommendations(req.title, cosine_sim)
    return {"recommended_movies": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
