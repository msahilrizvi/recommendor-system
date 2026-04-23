from src.recommend import recommend_collab, recommend_popular


class RecommendationService:
    def __init__(self, train_df, item_similarity_df):
        self.train_df = train_df
        self.item_similarity_df = item_similarity_df

    def get_recommendations(self, user_id, n=10):
        
        # Cold start handling
        if user_id not in self.train_df['user_id'].values:
            return {
                "user_id": user_id,
                "note": "User not found, returning popular items",
                "recommendations": recommend_popular(n)
            }

        recommendations = recommend_collab(
            user_id,
            self.train_df,
            self.item_similarity_df,
            n
        )

        return {
            "user_id": user_id,
            "recommendations": recommendations
        }