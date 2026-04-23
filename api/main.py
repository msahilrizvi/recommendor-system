from fastapi import FastAPI
from src.train import load_model
from src.service import RecommendationService
import pandas as pd
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

train_df = pd.read_csv(BASE_DIR / "data/processed/train.csv")
item_similarity_df = load_model()

service = RecommendationService(train_df, item_similarity_df)


@app.get("/")
def root():
    return {"message": "Recommendation API is running"}


@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    n = min(n, 20)
    return service.get_recommendations(user_id, n)