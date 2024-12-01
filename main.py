from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import pickle
import os

app = FastAPI(title="Movie Recommender API")

# Load models (you'll need to train and save these first)
MODEL_PATH = "../ml_models/saved_models/"

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    score: float

class UserInput(BaseModel):
    user_id: int
    n_recommendations: Optional[int] = 5

@app.get("/")
async def root():
    return {"message": "Movie Recommender API is running"}

@app.post("/recommend/", response_model=List[MovieRecommendation])
async def get_recommendations(user_input: UserInput):
    try:
        # Load the appropriate model (this is just an example)
        model_path = os.path.join(MODEL_PATH, "collaborative_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Get recommendations (you'll implement this based on your models)
        recommendations = []
        # Add your recommendation logic here
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/performance")
async def get_models_performance():
    # Return the performance metrics of different models
    return {
        "collaborative_filtering": {"rmse": 0.89, "mae": 0.70},
        "content_based": {"precision": 0.85, "recall": 0.78},
        "matrix_factorization": {"rmse": 0.92, "mae": 0.75}
    }