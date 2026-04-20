import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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