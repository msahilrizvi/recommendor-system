def recommend_collab(user_id, train_df, item_similarity_df, n=10):
    user_items = train_df[train_df['user_id'] == user_id]['item_id']
    
    scores = {}

    for item in user_items:
        similar_items = item_similarity_df[item]

        for sim_item, score in similar_items.items():
            if sim_item in user_items.values:
                continue

            if sim_item not in scores:
                scores[sim_item] = 0

            scores[sim_item] += score

    # Sort by score (descending)
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = [item for item, _ in ranked_items[:n]]

    return recommendations