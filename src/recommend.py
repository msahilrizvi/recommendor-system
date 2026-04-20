def get_similar_items(item_id, item_similarity_df, n=10):
    similar_items = item_similarity_df[item_id].sort_values(ascending=False)
    return similar_items.iloc[1:n+1].index.tolist()

def recommend_collab(user_id, train_df, item_similarity_df, n=10):
    user_items = train_df[train_df['user_id'] == user_id]['item_id']

    recommendations = []

    for item in user_items:
        similar_items = get_similar_items(item, item_similarity_df, n)
        recommendations.extend(similar_items)

    recommendations = list(set(recommendations))

    recommendations = [
        item for item in recommendations
        if item not in user_items.values
    ]

    return recommendations[:n]