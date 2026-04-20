from pathlib import Path
import pandas as pd
from src.train import train_model, save_model

BASE_DIR = Path(__file__).resolve().parent.parent
train_path = BASE_DIR / "data" / "processed" / "train.csv"

train_df = pd.read_csv(train_path)

item_similarity_df = train_model(train_df)
save_model(item_similarity_df)

print("Model trained and saved successfully!")