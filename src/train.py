from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Get project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "item_similarity.pkl"


def train_model(train_df):
    user_item_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)

    item_similarity = cosine_similarity(user_item_matrix.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return item_similarity_df

def save_model(item_similarity_df, path=MODEL_PATH):
    item_similarity_df.to_pickle(path)

def load_model(path=MODEL_PATH):
    return pd.read_pickle(path)