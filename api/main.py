from fastapi import FastAPI
from src.train import load_model
from src.recommend import recommend_collab
import pandas as pd
from pathlib import Path

app = FastAPI()

# Load data + model at startup
BASE_DIR = Path(__file__).resolve().parent.parent

train_df = pd.read_csv(BASE_DIR / "data/processed/train.csv")
item_similarity_df = load_model()


@app.get("/")
def root():
    return {"message": "Recommendation API is running"}


@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    recommendations = recommend_collab(user_id, train_df, item_similarity_df, n)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }