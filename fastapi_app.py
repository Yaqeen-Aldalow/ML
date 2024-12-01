# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# تعريف نموذج الطلب
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int

@app.post("/recommend/")
async def recommend_movies(req: RecommendationRequest):
    # هنا يمكن إضافة الكود الذي يقوم بتوليد التوصيات بناءً على user_id و n_recommendations
    # مثال افتراضي للأفلام الموصى بها:
    recommended_movies = [
        "Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5", 
        "Movie 6", "Movie 7", "Movie 8", "Movie 9", "Movie 10"
    ]
    
    # إرجاع التوصيات
    return {"movies": recommended_movies}
