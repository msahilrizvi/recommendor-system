def get_user_test_items(user_id, test_df):
    return test_df[test_df['user_id'] == user_id]['item_id'].values

def precision_recall_at_k(user_id, train_df, test_df, recommend_fn, item_similarity_df, K=10):
    
    recommended_items = recommend_fn(user_id, train_df, item_similarity_df, n=K)
    test_items = get_user_test_items(user_id, test_df)

    if len(test_items) == 0:
        return None, None

    recommended_set = set(recommended_items)
    test_set = set(test_items)

    relevant_items = recommended_set.intersection(test_set)

    precision = len(relevant_items) / K
    recall = len(relevant_items) / len(test_set)

    return precision, recall

def evaluate_model(train_df, test_df, recommend_fn, item_similarity_df, K=10):
    precisions = []
    recalls = []

    for user in test_df['user_id'].unique():
        p, r = precision_recall_at_k(user, train_df, test_df, recommend_fn, item_similarity_df, K)

        if p is not None:
            precisions.append(p)
            recalls.append(r)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    return avg_precision, avg_recall