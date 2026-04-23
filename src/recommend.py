def recommend_popular(train_df, n=10):
    """
    Recommend top-N most popular items based on interaction count.
    """
    item_popularity = train_df.groupby('item_id')['user_id'].count()
    item_popularity = item_popularity.sort_values(ascending=False)

    return item_popularity.head(n).index.tolist()


def recommend_collab(user_id, train_df, item_similarity_df, n=10):
    """
    Recommend items using item-based collaborative filtering
    with a scoring mechanism.
    """
    user_items = train_df[train_df['user_id'] == user_id]['item_id']

    scores = {}

    for item in user_items:
        if item not in item_similarity_df.columns:
            continue

        similar_items = item_similarity_df[item]

        for sim_item, score in similar_items.items():
            # Skip items already seen
            if sim_item in user_items.values:
                continue

            if sim_item not in scores:
                scores[sim_item] = 0

            scores[sim_item] += score

    # Sort items by score (descending)
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = [item for item, _ in ranked_items[:n]]

    return recommendations